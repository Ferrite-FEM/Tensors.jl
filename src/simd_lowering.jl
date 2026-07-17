# ========================================================================== #
#                                                                            #
#                               SIMD lowering                                #
#                                                                            #
# ========================================================================== #
#
# Two independent parts:
#
#  (1) Whole-data fast paths for `+`, `-`, scalar `*` and `/`, and `norm`:
#      load the full data tuple into one `SIMD.Vec`, operate, rebuild.
#      These replace the hand-written element-wise kernels of the old simd.jl.
#
#  (2) `try_simd_expr`, the vector lowering used by the einsum engine
#      (see einsum.jl for the pipeline): it inspects the *plan* of a binary
#      operation and, when the plan has "column structure" (explained at part
#      2 below), emits column-load/muladd kernels — the pattern the old
#      hand-written dot/dcontract/otimes kernels used. When the plan has no
#      such structure it returns `nothing` and the engine falls back to the
#      scalar lowering.
#
# Only same-eltype Float16/32/64 arguments take these paths; everything else
# (Dual numbers, integers, mixed eltypes, user types) uses the scalar
# expressions — matching the dispatch behavior of the old package.

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

# load a data tuple (or a contiguous range of it) into an SVec
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

# -------------------------------------------------------------------------- #
# (1) element-wise fast paths
# -------------------------------------------------------------------------- #
@inline function Base.:+(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        r = tosimd(S1.data) + tosimd(S2.data)
        return get_base(TT)(r)
    end
end
@inline function Base.:-(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        r = tosimd(S1.data) - tosimd(S2.data)
        return get_base(TT)(r)
    end
end
@inline function Base.:*(n::T, S::AllSIMDTensors{T}) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(n * tosimd(S.data))
end
@inline function Base.:*(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(tosimd(S.data) * n)
end
@inline function Base.:/(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds return get_base(typeof(S))(tosimd(S.data) / n)
end

# norm: orders 1 and 2 go through dot/dcontract (whose SIMD lowering comes
# from the engine); 4th order is a plain (weighted) sum of squares
@inline function LinearAlgebra.norm(S::Tensor{4, dim, T}) where {dim, T <: SIMDTypes}
    @inbounds begin
        SV = tosimd(S.data)
        return sqrt(sum(SV * SV))
    end
end
# mapreduce vectorizes better than the 81-wide SVec at this size
@inline LinearAlgebra.norm(S::Tensor{4, 3, T}) where {T <: SIMDTypes} = sqrt(mapreduce(abs2, +, S))
@generated function LinearAlgebra.norm(S::SymmetricTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    F = Expr(:tuple, [:(T($f)) for f in symmetric_multiplicities(4, dim)]...)
    M = n_components(SymmetricTensor{4, dim})
    return quote
        $(Expr(:meta, :inline))
        F = SVec{$M, T}($F)
        @inbounds begin
            SV = tosimd(S.data)
            return sqrt(sum(F * (SV * SV)))
        end
    end
end

# For hardware floats, contracting a mixed full/symmetric 4th-order pair is
# faster by densifying the symmetric argument and running the full-full
# kernel (the old package did the same); the engine's direct packed path
# stays in use for every other combination, where it is faster.
@inline function dcontract(S1::Tensor{4, dim, T}, S2::SymmetricTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2)
    return dcontract(SS1, SS2)
end
@inline function dcontract(S1::SymmetricTensor{4, dim, T}, S2::Tensor{4, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2)
    return dcontract(SS1, SS2)
end

# -------------------------------------------------------------------------- #
# (2) the engine's vector lowering
# -------------------------------------------------------------------------- #
#
# Input is the engine's plan for a binary operation: one `(products, mults)`
# pair per stored output component, where each product is a
# `(load index into A's data, load index into B's data)` pair (einsum.jl,
# step 3). The worked example here is `dot(A,B): C[i,j] = A[i,k]*B[k,j]`
# for two `Tensor{2,2}` — four output components, plans
#
#     C[1,1]: products [(1,1), (3,2)]      C[1,2]: products [(1,3), (3,4)]
#     C[2,1]: products [(2,1), (4,2)]      C[2,2]: products [(2,3), (4,4)]
#
# Reading the plans *by column of the output* (components 1..2 and 3..4),
# each column has the shape
#
#     row r, term t:   A-load = (base of term t) + r,   B-load fixed per term
#
# i.e. term t loads a contiguous run of A entries down the column — a column
# of the matrix A — and scales it with one B scalar. That is a vectorizable
# kernel: load each A-column once as an SVec, then chain muladds. For the
# example, `try_simd_columns` finds column height m = 2 and emits
#
#     SV1 = tosimd(_d1, Val(1), Val(2))     # A[:,1]
#     SV2 = tosimd(_d1, Val(3), Val(4))     # A[:,2]
#     c1 = muladd(SV2, _d2[2], SV1 * _d2[1])
#     c2 = muladd(SV2, _d2[4], SV1 * _d2[3])
#     return Tensor{2, 2}((c1[1], c1[2], c2[1], c2[2]))
#
# This is exactly the shape of the old hand-written kernels, generated from
# the plan instead of maintained per operation. Symmetric arguments fit the
# same pattern because their packed storage is itself column-major (e.g. the
# 6×6 packed grid of a SymmetricTensor{4,3}), with multiplicities folded into
# the B-side scalar (`_d2[k] * T(2)` — exact, the factor is a power of two).
#
# The numerics contract, precisely:
#
#   * Each output lane evaluates the plan's products in the same order as the
#     scalar lowering, one muladd per product — never regrouped into
#     `avec * (b1 + b2)`-style sums.
#   * The chains here always use muladd (as the old hand-written kernels
#     did), so for operations declared *without* `@muladd` (`dot`, `otimes`)
#     the hardware-float result may differ from the scalar path's `a*b + c*d`
#     in the last ulp where fma contracts. The scalar-output form below uses
#     a horizontal vector sum, again like the old kernels.
#   * In other words: the guarantee is not "SIMD ≡ scalar path" in all cases;
#     it is "each (operation, eltype) computes exactly what the pre-rewrite
#     package computed", with SIMD ≡ scalar in addition wherever the
#     declaration uses `@muladd` (the dcontract family).

# both arguments must be the same hardware float type
simd_eligible(TA, TB) = TA === TB && TA <: SIMDTypes

# the B-side scalar of one term: `_d2[ib]`, times the multiplicity if any
function simd_coef(db, ib, mult, T)
    return mult == 1 ? :($db[$ib]) : :($db[$ib] * $T($mult))
end

# Does the plan decompose into columns of height m? Output components are in
# storage order, so column J covers components (J-1)m+1 .. Jm. The column's
# first component is the reference: every further row r must have the same
# number of terms, the same B-side loads and multiplicities, and A-side loads
# shifted by exactly r (the contiguous run).
function has_column_structure(plans, m)
    ncols = length(plans) ÷ m
    for J in 1:ncols, r in 1:(m - 1)
        ref_products, ref_mults = plans[(J - 1) * m + 1]
        row_products, row_mults = plans[(J - 1) * m + 1 + r]
        length(row_products) == length(ref_products) || return false
        for t in eachindex(row_products)
            row_products[t][1] == ref_products[t][1] + r || return false # A-load: contiguous
            row_products[t][2] == ref_products[t][2]     || return false # B-load: fixed
            row_mults[t] == ref_mults[t]                 || return false
        end
    end
    return true
end

# Emit the column kernel, or return `nothing` when no column height works.
# Tries the tallest columns first (full-height loads vectorize best); the
# search is cheap and runs once per generated method.
function try_simd_columns(OutType, plans, da, db, T)
    ncomps = length(plans)
    for m in reverse(2:ncomps)
        ncomps % m == 0 || continue
        has_column_structure(plans, m) || continue
        ncols = ncomps ÷ m

        # each distinct A-run (identified by its start index) is loaded once
        loads = Int[]
        for J in 1:ncols, (ia, _) in plans[(J - 1) * m + 1][1]
            ia in loads || push!(loads, ia)
        end
        svname(ia) = Symbol(:SV, findfirst(==(ia), loads))
        stmts = Any[:($(svname(b)) = Tensors.tosimd($da, Val($b), Val($(b + m - 1)))) for b in loads]

        # one muladd chain per column, same term order as the scalar lowering
        nproducts = 0
        cols = Symbol[]
        for J in 1:ncols
            products, mults = plans[(J - 1) * m + 1]
            nproducts += m * length(products)
            acc = :($(svname(products[1][1])) * $(simd_coef(db, products[1][2], mults[1], T)))
            for t in 2:length(products)
                acc = :(muladd($(svname(products[t][1])), $(simd_coef(db, products[t][2], mults[t], T)), $acc))
            end
            c = Symbol(:c, J)
            push!(stmts, :($c = $acc))
            push!(cols, c)
        end

        # gather the column vectors into the output tuple
        out = Expr(:tuple)
        for c in cols, i in 1:m
            push!(out.args, :($c[$i]))
        end
        # The two largest kernels were deliberately not force-inlined in the
        # old package; keep that policy by size. 216 = 36 components × 6
        # terms, the 4s ⊡ 4s kernel at dim 3; the only other kernel that
        # reaches it is 4 ⊡ 4 at dim 3 (81 × 9 = 729). Everything smaller
        # (e.g. 4 ⊡ 2 at dim 3: 9 × 9 = 81) stays `@inline`.
        inline = nproducts < 216
        return quote
            $(stmts...)
            return $(OutType)($out)
        end, inline
    end
    return nothing
end

# Scalar output (a full contraction, e.g. `A[i,j]*B[i,j]`): when both sides
# read their entire data tuples in storage order, this is an element-wise
# product followed by a horizontal sum, with the symmetric multiplicities as
# a constant weight vector.
function try_simd_scalar(plans, da, db, T)
    length(plans) == 1 || return nothing
    products, mults = plans[1]
    M = length(products)
    M > 1 || return nothing
    all(t -> products[t] == (t, t), 1:M) || return nothing
    if all(==(1), mults)
        return quote
            SV1 = Tensors.tosimd($da); SV2 = Tensors.tosimd($db)
            return sum(SV1 * SV2)
        end
    else
        F = Expr(:tuple, [:($T($f)) for f in mults]...)
        return quote
            F = Tensors.SVec{$M, $T}($F)
            SV1 = Tensors.tosimd($da); SV2 = Tensors.tosimd($db)
            return sum(F * (SV1 * SV2))
        end
    end
end

# Entry point called by the engine. Returns `nothing` (no vectorizable
# structure — use the scalar lowering) or `(expr, inline::Bool)`.
#
# `da`/`db` are the variable *names* the kernel reads from (`:_d1`, `:_d2`);
# `dataA`/`dataB` are the expressions that initialize them.
#
# Loads always come from the *first* argument: the declarations place the
# argument that carries the leading output indices first, which is what makes
# its loads contiguous. Plans without that property (e.g. `dot(v, A)`, where
# the reads of `A` are strided) simply fall back to the scalar lowering —
# also what the old hand-written kernels did.
function try_simd_expr(OutType, plans, da, db, dataA, dataB, T)
    inline = true
    if OutType === nothing
        core = try_simd_scalar(plans, da, db, T)
        core === nothing && return nothing
    else
        r = try_simd_columns(OutType, plans, da, db, T)
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
