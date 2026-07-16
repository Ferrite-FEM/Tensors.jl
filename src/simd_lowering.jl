####################################################################
# SIMD lowering                                                    #
#                                                                  #
# Two parts:                                                       #
#  1. Whole-data fast paths for +, -, scalar *, / and norm using   #
#     SIMD.Vec (replaces the hand-written kernels of the old       #
#     simd.jl for element-wise operations).                        #
#  2. `try_simd_expr` — called by the einsum engine: detects the   #
#     packed-space column structure of a planned contraction and   #
#     emits column-load/muladd kernels (the pattern the old        #
#     hand-written dot/dcontract/otimes kernels used). Returns     #
#     `nothing` when the plan has no such structure, in which case #
#     the scalar lowerer is used.                                  #
#                                                                  #
# Only same-eltype Float16/32/64 arguments take these paths; all   #
# other eltypes use the scalar expressions (matching the old       #
# package's dispatch reality).                                     #
####################################################################

import SIMD
const SVec{N, T} = SIMD.Vec{N, T}

const SIMDTypes = Union{Float16, Float32, Float64}

# Tensors which the whole-data element-wise fast paths apply to.
# Note: Tensor{4,3} (81 components) is deliberately excluded, as in the old
# simd.jl — LLVM does better on the scalar code for that width.
const AllSIMDTensors{T <: SIMDTypes} = Union{
    Tensor{1, 1, T, 1}, Tensor{1, 2, T, 2}, Tensor{1, 3, T, 3},
    Tensor{2, 1, T, 1}, Tensor{2, 2, T, 4}, Tensor{2, 3, T, 9},
    Tensor{4, 1, T, 1}, Tensor{4, 2, T, 16}, #= Tensor{4, 3, T, 81}, =#
    SymmetricTensor{2, 1, T, 1}, SymmetricTensor{2, 2, T, 3}, SymmetricTensor{2, 3, T, 6},
    SymmetricTensor{4, 1, T, 1}, SymmetricTensor{4, 2, T, 9}, SymmetricTensor{4, 3, T, 36}}

# load a data tuple (or a range of it) into an SVec
@inline tosimd(D::NTuple{N, T}) where {N, T} = SVec{N, T}(D)

@generated function tosimd(D::NTuple{N, T}, ::Val{strt}, ::Val{stp}) where {N, T, strt, stp}
    expr = Expr(:tuple, [:(D[$i]) for i in strt:stp]...)
    M = stp - strt + 1
    return quote
        $(Expr(:meta, :inline))
        @inbounds return SVec{$M, T}($expr)
    end
end

# build a tensor from one SVec
# (two separate methods, like the old simd.jl, to dispatch correctly and to
# avoid ambiguities with the default and Vararg constructors)
@generated function (::Type{Tensor{order, dim}})(r::SVec{N, T}) where {order, dim, N, T}
    return quote
        $(Expr(:meta, :inline))
        Tensor{order, dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end
@generated function (::Type{SymmetricTensor{order, dim}})(r::SVec{N, T}) where {order, dim, N, T}
    return quote
        $(Expr(:meta, :inline))
        SymmetricTensor{order, dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end

# multiplicity of each stored component of a symmetric tensor (how many
# components of the corresponding full tensor map to it): 1, 2 or 4.
function symmetric_multiplicities(order::Int, dim::Int)
    counts = zeros(Int, n_components(SymmetricTensor{order, dim}))
    for c in base_components(Tensor{order, dim})
        counts[compute_index(SymmetricTensor{order, dim}, c...)] += 1
    end
    return counts
end

################################
# (1) element-wise fast paths  #
################################
@inline function Base.:+(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        r = tosimd(get_data(S1)) + tosimd(get_data(S2))
        return get_base(TT)(r)
    end
end
@inline function Base.:-(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        r = tosimd(get_data(S1)) - tosimd(get_data(S2))
        return get_base(TT)(r)
    end
end
@inline function Base.:*(n::T, S::AllSIMDTensors{T}) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(n * tosimd(get_data(S)))
end
@inline function Base.:*(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(tosimd(get_data(S)) * n)
end
@inline function Base.:/(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(tosimd(get_data(S)) / n)
end

########################
# (1b) norm fast paths #
########################
# order 1 and 2 rely on dot/dcontract (whose SIMD lowering comes from the engine)
@inline function LinearAlgebra.norm(S::Tensor{4, dim, T}) where {dim, T <: SIMDTypes}
    @inbounds begin
        SV = tosimd(get_data(S))
        return sqrt(sum(SV * SV))
    end
end
@generated function LinearAlgebra.norm(S::SymmetricTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    F = Expr(:tuple, [:(T($f)) for f in symmetric_multiplicities(4, dim)]...)
    M = n_components(SymmetricTensor{4, dim})
    return quote
        $(Expr(:meta, :inline))
        F = SVec{$M, T}($F)
        @inbounds begin
            SV = tosimd(get_data(S))
            return sqrt(sum(F * (SV * SV)))
        end
    end
end

###################################
# (1c) mixed 4th-order fast paths #
###################################
# For hardware floats, contracting a mixed full/symmetric 4th-order pair is
# faster by densifying the symmetric argument and running the full-full kernel
# (the old package did the same); the engine's direct packed path stays in use
# for every other combination, where it is faster.
@inline function dcontract(S1::Tensor{4, dim, T}, S2::SymmetricTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2)
    return dcontract(SS1, SS2)
end
@inline function dcontract(S1::SymmetricTensor{4, dim, T}, S2::Tensor{4, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2)
    return dcontract(SS1, SS2)
end

##############################
# (2) engine SIMD lowering   #
##############################
# `plans` is a vector over output components of (uniq, factors) as produced by
# `component_products`. Detect column structure and emit an SVec kernel, or
# return `nothing` to fall back to scalar lowering.

simd_eligible(TA, TB) = TA === TB && TA <: SIMDTypes

# coefficient expression for one term: factor * db[ib], with the eltype-exact
# factor folding of the old kernels (`db[k] * T(2)`; the factor is a small
# power of two, so this is exact)
function simd_coef(db, ib, factor, T)
    return factor == 1 ? :($db[$ib]) : :($db[$ib] * $T($factor))
end

# Try to lower with contiguous loads from `da` (columns of A) scaled by scalars
# from `db`. `plans[n]` must decompose so that within an output column of
# height m, term t reads da[base_t + r] with shared (ib, factor).
#
# NOTE: terms are emitted one muladd per (ia, ib) product in the plan's order —
# never merged into `avec * (b1 + b2)` — so each output lane evaluates the
# exact same muladd chain as the scalar lowering (IEEE-identical).
function try_simd_columns(OutType, plans, da, db, T)
    N = length(plans)
    for m in reverse(2:N)
        N % m == 0 || continue
        ncols = N ÷ m
        ok = true
        for J in 1:ncols, r in 1:(m - 1)
            base = plans[(J - 1) * m + 1]
            cur = plans[(J - 1) * m + 1 + r]
            if length(cur[1]) != length(base[1]) ||
               any(t -> cur[1][t][2] != base[1][t][2] || cur[2][t] != base[2][t] ||
                        cur[1][t][1] != base[1][t][1] + r, eachindex(cur[1]))
                ok = false; break
            end
        end
        ok || continue
        # distinct A-columns (by base load index), in first-appearance order
        loads = Int[]
        for J in 1:ncols, (ia, _) in plans[(J - 1) * m + 1][1]
            ia in loads || push!(loads, ia)
        end
        svname(ia) = Symbol(:SV, findfirst(==(ia), loads))
        stmts = Any[:($(svname(b)) = Tensors.tosimd($da, Val($b), Val($(b + m - 1)))) for b in loads]
        nproducts = 0
        cols = Symbol[]
        for J in 1:ncols
            uniq, factors = plans[(J - 1) * m + 1]
            nproducts += m * length(uniq)
            acc = :($(svname(uniq[1][1])) * $(simd_coef(db, uniq[1][2], factors[1], T)))
            for t in 2:length(uniq)
                acc = :(muladd($(svname(uniq[t][1])), $(simd_coef(db, uniq[t][2], factors[t], T)), $acc))
            end
            c = Symbol(:c, J)
            push!(stmts, :($c = $acc))
            push!(cols, c)
        end
        out = Expr(:tuple)
        for c in cols, i in 1:m
            push!(out.args, :($c[$i]))
        end
        # the two largest kernels (4-4 and 4s-4s at dim 3) were deliberately not
        # force-inlined in the old package; keep that policy by a size threshold
        inline = nproducts < 216
        return quote
            $(stmts...)
            return $(OutType)($out)
        end, inline
    end
    return nothing
end

# scalar-output form: full-data element-wise product (+ symmetric weights)
function try_simd_scalar(plans, da, db, T)
    length(plans) == 1 || return nothing
    uniq, factors = plans[1]
    M = length(uniq)
    M > 1 || return nothing
    all(t -> uniq[t] == (t, t), 1:M) || return nothing
    if all(==(1), factors)
        return quote
            SV1 = Tensors.tosimd($da); SV2 = Tensors.tosimd($db)
            return sum(SV1 * SV2)
        end
    else
        F = Expr(:tuple, [:($T($f)) for f in factors]...)
        return quote
            F = Tensors.SVec{$M, $T}($F)
            SV1 = Tensors.tosimd($da); SV2 = Tensors.tosimd($db)
            return sum(F * (SV1 * SV2))
        end
    end
end

# Returns `nothing`, or `(expr, inline::Bool)`.
function try_simd_expr(OutType, plans, da, db, dataA, dataB, T)
    inline = true
    if OutType === nothing
        core = try_simd_scalar(plans, da, db, T)
        core === nothing && return nothing
    else
        r = try_simd_columns(OutType, plans, da, db, T)
        if r === nothing
            # try the swapped orientation: contiguous loads from B
            splans = [(map(reverse, uniq), factors) for (uniq, factors) in plans]
            r = try_simd_columns(OutType, splans, db, da, T)
        end
        r === nothing && return nothing
        core, inline = r
    end
    return quote
        $da = $dataA; $db = $dataB
        @inbounds begin
            $core
        end
    end, inline
end
