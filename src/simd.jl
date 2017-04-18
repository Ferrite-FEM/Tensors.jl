#=
This file contains explicit SIMD instructions for tensors.
Many of the methods defined outside this file will use SIMD-instructions
if julia is ran with -O3. Even if -O3 is enabled, the compiler is sometimes
thrown off guard, and therefore, explicit SIMD routines are
defined. This will enable SIMD-instructions even if julia is ran with
the default -O2.

The functions here are only defined for tensors of the same
element type. Otherwise it does work. Promotion should take
care of this before the tensors enter the functions here.

The file is organized as follows:
(1): + and - between tensors
(2): * and / between tensor and number
(3): dot
(4): dcontract
(5): otimes
(6): norm
=#

import SIMD
const SVec{N, T} = SIMD.Vec{N, T}

const SIMDTypes = Union{Float16, Float32, Float64}

const AllSIMDTensors{T <: SIMDTypes} = Union{Tensor{1, 1, T, 1}, Tensor{1, 2, T, 2}, Tensor{1, 3, T, 3},
                                             Tensor{2, 1, T, 1}, Tensor{2, 2, T, 4}, Tensor{2, 3, T, 9},
                                             Tensor{4, 1, T, 1}, Tensor{4, 2, T, 16}, #=Tensor{4, 3, T, 81},=#
                                             SymmetricTensor{2, 1, T, 1}, SymmetricTensor{2, 2, T, 3}, SymmetricTensor{2, 3, T, 6},
                                             SymmetricTensor{4, 1, T, 1}, SymmetricTensor{4, 2, T, 9}, SymmetricTensor{4, 3, T, 36}}

# SIMD sizes accepted by LLVM between 1 and 100
const SIMD_CHUNKS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 65, 66, 68, 72, 80, 96)

# factors for the symmetric tensors, return a quote
function symmetric_factors(order, dim, T)
    if order == 2
        dim == 1 && return :(SVec{1, $T}(($T(1),)))
        dim == 2 && return :(SVec{3, $T}(($T(1),$T(2),$T(1))))
        dim == 3 && return :(SVec{6, $T}(($T(1),$T(2),$T(2),$T(1),$T(2),$T(1))))
    elseif order == 4
        dim == 1 && return :(SVec{1, $T}(($T(1),)))
        dim == 2 && return :(SVec{9, $T}(($T(1),$T(2),$T(1),$T(2),$T(4),$T(2),$T(1),$T(2),$T(1))))
        dim == 3 && return :(SVec{36,$T}(($T(1),$T(2),$T(2),$T(1),$T(2),$T(1),$T(2),$T(4),$T(4),$T(2),$T(4),$T(2),
                                          $T(2),$T(4),$T(4),$T(2),$T(4),$T(2),$T(1),$T(2),$T(2),$T(1),$T(2),$T(1),
                                          $T(2),$T(4),$T(4),$T(2),$T(4),$T(2),$T(1),$T(2),$T(2),$T(1),$T(2),$T(1))))
    end
end

# Tensor from SVec (Note: This needs to be two separate methods in order to dispatch correctly)
@generated function (::Type{Tensor{order, dim}})(r::SVec{N, T}) where {order, dim, N, T}
    return quote
        $(Expr(:meta, :inline))
        Tensor{$order, $dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end
@generated function (::Type{SymmetricTensor{order, dim}})(r::SVec{N, T}) where {order, dim, N, T}
    return quote
        $(Expr(:meta, :inline))
        SymmetricTensor{$order, $dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end
# constructor from several SVecs which happens in dot, dcontract and otimes
@generated function (::Type{Tensor{order, dim}})(r::NTuple{M, SVec{N, T}}) where {order, dim, M, N, T}
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{$order, $dim}($(Expr(:tuple, [:(r[$j][$i]) for i in 1:N, j in 1:M]...)))
    end
end
@generated function (::Type{SymmetricTensor{order, dim}})(r::NTuple{M, SVec{N, T}}) where {order, dim, M, N, T}
    return quote
        $(Expr(:meta, :inline))
        @inbounds return SymmetricTensor{$order, $dim}($(Expr(:tuple, [:(r[$j][$i]) for i in 1:N, j in 1:M]...)))
    end
end

# load Tensor into SVec
@inline tosimd(A::NTuple{N, T}) where {N, T} = SVec{N, T}(A)

# load given range of linear indices into SVec
@generated function tosimd(D::NTuple{N, T}, ::Type{Val{strt}}, ::Type{Val{stp}}) where {N, T, strt, stp}
    expr = Expr(:tuple, [:(D[$i]) for i in strt:stp]...)
    M = length(expr.args)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return SVec{$M, T}($expr)
    end
end

################################
# (1): + and - between tensors #
################################
@inline function Base.:+(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        D1 = get_data(S1); SV1 = tosimd(D1)
        D2 = get_data(S2); SV2 = tosimd(D2)
        r = SV1 + SV2
        return get_base(TT)(r)
    end
end
@generated function Base.:+(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T}) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = get_data(S1); SV180 = tosimd(D1, Val{1}, Val{80})
            D2 = get_data(S2); SV280 = tosimd(D2, Val{1}, Val{80})
            r = SV180 + SV280
            r81 = D1[81] + D2[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@inline function Base.:-(S1::TT, S2::TT) where {TT <: AllSIMDTensors}
    @inbounds begin
        D1 = get_data(S1); SV1 = tosimd(D1)
        D2 = get_data(S2); SV2 = tosimd(D2)
        r = SV1 - SV2
        return get_base(TT)(r)
    end
end
@generated function Base.:-(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T}) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = get_data(S1); SV180 = tosimd(D1, Val{1}, Val{80})
            D2 = get_data(S2); SV280 = tosimd(D2, Val{1}, Val{80})
            r = SV180 - SV280
            r81 = D1[81] - D2[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end

##########################################
# (2): * and / between tensor and number #
##########################################
@inline function Base.:*(n::T, S::AllSIMDTensors{T}) where {T <: SIMDTypes}
    @inbounds begin
        D = get_data(S); SV = tosimd(D)
        r = n * SV
        return get_base(typeof(S))(r)
    end
end
@inline function Base.:*(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds begin
        D = get_data(S); SV = tosimd(D)
        r = SV * n
        return get_base(typeof(S))(r)
    end
end
@inline function Base.:/(S::AllSIMDTensors{T}, n::T) where {T <: SIMDTypes}
    @inbounds begin
        D = get_data(S); SV = tosimd(D)
        r = SV / n
        return get_base(typeof(S))(r)
    end
end
@generated function Base.:*(n::T, S::Tensor{4, 3, T}) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S); SV80 = tosimd(D, Val{1}, Val{80})
            r = n * SV80; r81 = n * D[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:*(S::Tensor{4, 3, T}, n::T) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S); SV80 = tosimd(D, Val{1}, Val{80})
            r = SV80 * n; r81 = D[81] * n
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:/(S::Tensor{4, 3, T}, n::T) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S); SV80 = tosimd(D, Val{1}, Val{80})
            r = SV80 / n; r81 = D[81] / n
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end

############
# (3): dot #
############
# 2s-1
@inline Base.dot(S1::SymmetricTensor{2, dim, T}, S2::Vec{dim, T}) where {dim, T <: SIMDTypes} = dot(promote(S1), S2)
# 2-1
@inline function Base.dot(S1::Tensor{2, 2, T}, S2::Vec{2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1}, Val{2})
        SV12 = tosimd(D1, Val{3}, Val{4})
        r = muladd(SV12, D2[2], SV11 * D2[1])
        return Tensor{1, 2}(r)
    end
end
@inline function Base.dot(S1::Tensor{2, 3, T}, S2::Vec{3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1}, Val{3})
        SV12 = tosimd(D1, Val{4}, Val{6})
        SV13 = tosimd(D1, Val{7}, Val{9})
        r = muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))
        return Tensor{1, 3}(r)
    end
end

# 2s-2 / 2-2s
@inline function Base.dot(S1::AbstractTensor{2, dim, T}, S2::AbstractTensor{2, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2); dot(SS1, SS2)
end
# 2-2
@inline function Base.dot(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1}, Val{2})
        SV12 = tosimd(D1, Val{3}, Val{4})
        r1 = muladd(SV12, D2[2], SV11 * D2[1])
        r2 = muladd(SV12, D2[4], SV11 * D2[3])
        return Tensor{2, 2}((r1, r2))
    end
end
@inline function Base.dot(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1}, Val{3})
        SV12 = tosimd(D1, Val{4}, Val{6})
        SV13 = tosimd(D1, Val{7}, Val{9})
        r1 = muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))
        r2 = muladd(SV13, D2[6], muladd(SV12, D2[5], SV11 * D2[4]))
        r3 = muladd(SV13, D2[9], muladd(SV12, D2[8], SV11 * D2[7]))
        return Tensor{2, 3}((r1, r2, r3))
    end
end

##################
# (4): dcontract #
##################
# 2s-2 / 2-2s
@inline function dcontract(S1::AbstractTensor{2, dim, T}, S2::AbstractTensor{2, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2); dcontract(SS1, SS2)
end
# 2-2
@inline function dcontract(S1::Tensor{2, dim, T}, S2::Tensor{2, dim, T}) where {dim, T <: SIMDTypes}
    SV1 = tosimd(get_data(S1))
    SV2 = tosimd(get_data(S2))
    r = SV1 * SV2
    return sum(r)
end
# 2s-2s
@generated function dcontract(S1::SymmetricTensor{2, dim, T}, S2::SymmetricTensor{2, dim, T}) where {dim, T <: SIMDTypes}
    F = symmetric_factors(2, dim, T)
    return quote
        $(Expr(:meta, :inline))
        F = $F
        SV1 = tosimd(get_data(S1))
        SV2 = tosimd(get_data(S2))
        SV1SV2 = SV1 * SV2; FSV1SV2 = F * SV1SV2
        return sum(FSV1SV2)
    end
end

# 4-2s
@inline dcontract(S1::Tensor{4, 3, T}, S2::SymmetricTensor{2, 3, T}) where {T <: SIMDTypes} = dcontract(S1, promote(S2))
# 4-2
@inline function dcontract(S1::Tensor{4, 2, T}, S2::Tensor{2, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1}, Val{4})
        SV12 = tosimd(D1, Val{5}, Val{8})
        SV13 = tosimd(D1, Val{9}, Val{12})
        SV14 = tosimd(D1, Val{13}, Val{16})
        r = muladd(SV14, D2[4], muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1])))
        return Tensor{2, 2}(r)
    end
end
@inline function dcontract(S1::Tensor{4, 3, T}, S2::Tensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{9})
        SV12 = tosimd(D1, Val{10}, Val{18})
        SV13 = tosimd(D1, Val{19}, Val{27})
        SV14 = tosimd(D1, Val{28}, Val{36})
        SV15 = tosimd(D1, Val{37}, Val{45})
        SV16 = tosimd(D1, Val{46}, Val{54})
        SV17 = tosimd(D1, Val{55}, Val{63})
        SV18 = tosimd(D1, Val{64}, Val{72})
        SV19 = tosimd(D1, Val{73}, Val{81})
        r = muladd(SV19, D2[9], muladd(SV18, D2[8], muladd(SV17, D2[7], muladd(SV16, D2[6], muladd(SV15, D2[5], muladd(SV14, D2[4], muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))))))))
        return Tensor{2, 3}(r)
    end
end
@inline function dcontract(S1::SymmetricTensor{4, 3, T}, S2::SymmetricTensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{6})
        SV12 = tosimd(D1, Val{7},  Val{12})
        SV13 = tosimd(D1, Val{13}, Val{18})
        SV14 = tosimd(D1, Val{19}, Val{24})
        SV15 = tosimd(D1, Val{25}, Val{30})
        SV16 = tosimd(D1, Val{31}, Val{36})
        D21 = D2[1]; D22 = D2[2] * T(2); D23 = D2[3] * T(2)
        D24 = D2[4]; D25 = D2[5] * T(2); D26 = D2[6]
        r = muladd(SV16, D26, muladd(SV15, D25, muladd(SV14, D24, muladd(SV13, D23, muladd(SV12, D22, SV11 * D21)))))
        return SymmetricTensor{2, 3}(r)
    end
end

# 4s-4 / 4-4s
@inline function dcontract(S1::AbstractTensor{4, dim, T}, S2::AbstractTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2); dcontract(SS1, SS2)
end
# 4-4
@inline function dcontract(S1::Tensor{4, 2, T}, S2::Tensor{4, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{4})
        SV12 = tosimd(D1, Val{5},  Val{8})
        SV13 = tosimd(D1, Val{9},  Val{12})
        SV14 = tosimd(D1, Val{13}, Val{16})
        r1 = muladd(SV14, D2[4],  muladd(SV13, D2[3],  muladd(SV12, D2[2],  SV11 * D2[1])))
        r2 = muladd(SV14, D2[8],  muladd(SV13, D2[7],  muladd(SV12, D2[6],  SV11 * D2[5])))
        r3 = muladd(SV14, D2[12], muladd(SV13, D2[11], muladd(SV12, D2[10], SV11 * D2[9])))
        r4 = muladd(SV14, D2[16], muladd(SV13, D2[15], muladd(SV12, D2[14], SV11 * D2[13])))
        return Tensor{4, 2}((r1, r2, r3, r4))
    end
end
function dcontract(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{9})
        SV12 = tosimd(D1, Val{10}, Val{18})
        SV13 = tosimd(D1, Val{19}, Val{27})
        SV14 = tosimd(D1, Val{28}, Val{36})
        SV15 = tosimd(D1, Val{37}, Val{45})
        SV16 = tosimd(D1, Val{46}, Val{54})
        SV17 = tosimd(D1, Val{55}, Val{63})
        SV18 = tosimd(D1, Val{64}, Val{72})
        SV19 = tosimd(D1, Val{73}, Val{81})
        r1 = muladd(SV19, D2[9],  muladd(SV18, D2[8],  muladd(SV17, D2[7],  muladd(SV16, D2[6],  muladd(SV15, D2[5],  muladd(SV14, D2[4],  muladd(SV13, D2[3],  muladd(SV12, D2[2],  SV11 * D2[1] ))))))))
        r2 = muladd(SV19, D2[18], muladd(SV18, D2[17], muladd(SV17, D2[16], muladd(SV16, D2[15], muladd(SV15, D2[14], muladd(SV14, D2[13], muladd(SV13, D2[12], muladd(SV12, D2[11], SV11 * D2[10]))))))))
        r3 = muladd(SV19, D2[27], muladd(SV18, D2[26], muladd(SV17, D2[25], muladd(SV16, D2[24], muladd(SV15, D2[23], muladd(SV14, D2[22], muladd(SV13, D2[21], muladd(SV12, D2[20], SV11 * D2[19]))))))))
        r4 = muladd(SV19, D2[36], muladd(SV18, D2[35], muladd(SV17, D2[34], muladd(SV16, D2[33], muladd(SV15, D2[32], muladd(SV14, D2[31], muladd(SV13, D2[30], muladd(SV12, D2[29], SV11 * D2[28]))))))))
        r5 = muladd(SV19, D2[45], muladd(SV18, D2[44], muladd(SV17, D2[43], muladd(SV16, D2[42], muladd(SV15, D2[41], muladd(SV14, D2[40], muladd(SV13, D2[39], muladd(SV12, D2[38], SV11 * D2[37]))))))))
        r6 = muladd(SV19, D2[54], muladd(SV18, D2[53], muladd(SV17, D2[52], muladd(SV16, D2[51], muladd(SV15, D2[50], muladd(SV14, D2[49], muladd(SV13, D2[48], muladd(SV12, D2[47], SV11 * D2[46]))))))))
        r7 = muladd(SV19, D2[63], muladd(SV18, D2[62], muladd(SV17, D2[61], muladd(SV16, D2[60], muladd(SV15, D2[59], muladd(SV14, D2[58], muladd(SV13, D2[57], muladd(SV12, D2[56], SV11 * D2[55]))))))))
        r8 = muladd(SV19, D2[72], muladd(SV18, D2[71], muladd(SV17, D2[70], muladd(SV16, D2[69], muladd(SV15, D2[68], muladd(SV14, D2[67], muladd(SV13, D2[66], muladd(SV12, D2[65], SV11 * D2[64]))))))))
        r9 = muladd(SV19, D2[81], muladd(SV18, D2[80], muladd(SV17, D2[79], muladd(SV16, D2[78], muladd(SV15, D2[77], muladd(SV14, D2[76], muladd(SV13, D2[75], muladd(SV12, D2[74], SV11 * D2[73]))))))))
        return Tensor{4, 3}((r1, r2, r3, r4, r5, r6, r7, r8, r9))
    end
end
@inline function dcontract(S1::SymmetricTensor{4, 3, T}, S2::Tensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{6})
        SV12 = tosimd(D1, Val{7}, Val{12})
        SV13 = tosimd(D1, Val{13}, Val{18})
        SV14 = tosimd(D1, Val{19}, Val{24})
        SV15 = tosimd(D1, Val{25}, Val{30})
        SV16 = tosimd(D1, Val{31}, Val{36})
        D21 = D2[1]; D22 = D2[2] + D2[4]; D23 = D2[3] + D2[7]
        D24 = D2[5]; D25 = D2[6] + D2[8]; D26 = D2[9]
        r = muladd(SV16, D26, muladd(SV15, D25, muladd(SV14, D24, muladd(SV13, D23, muladd(SV12, D22, SV11 * D21)))))
        return SymmetricTensor{2, 3}(r)
    end
end
@inline function dcontract(S1::SymmetricTensor{4, 2, T}, S2::SymmetricTensor{4, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{3})
        SV12 = tosimd(D1, Val{4},  Val{6})
        SV13 = tosimd(D1, Val{7},  Val{9})
        D21  = D2[1]; D22 = D2[2] * T(2); D23 = D2[3]
        D24  = D2[4]; D25 = D2[5] * T(2); D26 = D2[6]
        D27  = D2[7]; D28 = D2[8] * T(2); D29 = D2[9]
        r1 = muladd(SV13, D23, muladd(SV12, D22, SV11 * D21))
        r2 = muladd(SV13, D26, muladd(SV12, D25, SV11 * D24))
        r3 = muladd(SV13, D29, muladd(SV12, D28, SV11 * D27))
        return SymmetricTensor{4, 2}((r1, r2, r3))
    end
end
function dcontract(S1::SymmetricTensor{4, 3, T}, S2::SymmetricTensor{4, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV11 = tosimd(D1, Val{1},  Val{6})
        SV12 = tosimd(D1, Val{7},  Val{12})
        SV13 = tosimd(D1, Val{13}, Val{18})
        SV14 = tosimd(D1, Val{19}, Val{24})
        SV15 = tosimd(D1, Val{25}, Val{30})
        SV16 = tosimd(D1, Val{31}, Val{36})
        D21  = D2[1];  D22  = D2[2]  * T(2); D23  = D2[3]  * T(2); D24  = D2[4];  D25  = D2[5]  * T(2); D26  = D2[6]
        D27  = D2[7];  D28  = D2[8]  * T(2); D29  = D2[9]  * T(2); D210 = D2[10]; D211 = D2[11] * T(2); D212 = D2[12]
        D213 = D2[13]; D214 = D2[14] * T(2); D215 = D2[15] * T(2); D216 = D2[16]; D217 = D2[17] * T(2); D218 = D2[18]
        D219 = D2[19]; D220 = D2[20] * T(2); D221 = D2[21] * T(2); D222 = D2[22]; D223 = D2[23] * T(2); D224 = D2[24]
        D225 = D2[25]; D226 = D2[26] * T(2); D227 = D2[27] * T(2); D228 = D2[28]; D229 = D2[29] * T(2); D230 = D2[30]
        D231 = D2[31]; D232 = D2[32] * T(2); D233 = D2[33] * T(2); D234 = D2[34]; D235 = D2[35] * T(2); D236 = D2[36]
        r1 = muladd(SV16, D26,  muladd(SV15, D25,  muladd(SV14, D24,  muladd(SV13, D23,  muladd(SV12, D22,  SV11 * D21 )))))
        r2 = muladd(SV16, D212, muladd(SV15, D211, muladd(SV14, D210, muladd(SV13, D29,  muladd(SV12, D28,  SV11 * D27 )))))
        r3 = muladd(SV16, D218, muladd(SV15, D217, muladd(SV14, D216, muladd(SV13, D215, muladd(SV12, D214, SV11 * D213)))))
        r4 = muladd(SV16, D224, muladd(SV15, D223, muladd(SV14, D222, muladd(SV13, D221, muladd(SV12, D220, SV11 * D219)))))
        r5 = muladd(SV16, D230, muladd(SV15, D229, muladd(SV14, D228, muladd(SV13, D227, muladd(SV12, D226, SV11 * D225)))))
        r6 = muladd(SV16, D236, muladd(SV15, D235, muladd(SV14, D234, muladd(SV13, D233, muladd(SV12, D232, SV11 * D231)))))
        return SymmetricTensor{4, 3}((r1, r2, r3, r4, r5, r6))
    end
end

###############
# (5): otimes #
###############
# 1-1
@inline function otimes(S1::Vec{2, T}, S2::Vec{2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]
        r2 = SV1 * D2[2]
        return Tensor{2, 2}((r1, r2))
    end
end
@inline function otimes(S1::Vec{3, T}, S2::Vec{3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]; r2 = SV1 * D2[2]; r3 = SV1 * D2[3]
        return Tensor{2, 3}((r1, r2, r3))
    end
end

# 2s-2 / 2-2s
@inline function otimes(S1::AbstractTensor{2, dim, T}, S2::AbstractTensor{2, dim, T}) where {dim, T <: SIMDTypes}
    SS1, SS2 = promote_base(S1, S2); otimes(SS1, SS2)
end

# 2-2
@inline function otimes(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]; r2 = SV1 * D2[2]; r3 = SV1 * D2[3]; r4 = SV1 * D2[4]
        return Tensor{4, 2}((r1, r2, r3, r4))
    end
end
@inline function otimes(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]; r2 = SV1 * D2[2]; r3 = SV1 * D2[3]
        r4 = SV1 * D2[4]; r5 = SV1 * D2[5]; r6 = SV1 * D2[6]
        r7 = SV1 * D2[7]; r8 = SV1 * D2[8]; r9 = SV1 * D2[9]
        return Tensor{4, 3}((r1, r2, r3, r4, r5, r6, r7, r8, r9))
    end
end

# 2s-2s
@inline function otimes(S1::SymmetricTensor{2, 2, T}, S2::SymmetricTensor{2, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]; r2 = SV1 * D2[2]; r3 = SV1 * D2[3]
        return SymmetricTensor{4, 2}((r1, r2, r3))
    end
end
@inline function otimes(S1::SymmetricTensor{2, 3, T}, S2::SymmetricTensor{2, 3, T}) where {T <: SIMDTypes}
    @inbounds begin
        D1 = get_data(S1); D2 = get_data(S2)
        SV1 = tosimd(D1)
        r1 = SV1 * D2[1]; r2 = SV1 * D2[2]; r3 = SV1 * D2[3]
        r4 = SV1 * D2[4]; r5 = SV1 * D2[5]; r6 = SV1 * D2[6]
        return SymmetricTensor{4, 3}((r1, r2, r3, r4, r5, r6))
    end
end

#############
# (6): norm #
#############
# order 1 and order 2 norms rely on dot and dcontract respectively
@inline function Base.norm(S::Tensor{4, 2, T}) where {T <: SIMDTypes}
    @inbounds begin
        SV = tosimd(get_data(S))
        SVSV = SV * SV
        r = sum(SVSV)
        return sqrt(r)
    end
end
@generated function Base.norm(S::Tensor{4, 3, T}) where {T <: SIMDTypes}
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            SV80 = tosimd(D, Val{1}, Val{80})
            SV80SV80 = SV80 * SV80
            r80 = sum(SV80SV80)
            r = r80 + D[81] * D[81]
            return sqrt(r)
        end
    end
end
@generated function Base.norm(S::SymmetricTensor{4, dim, T}) where {dim, T <: SIMDTypes}
    F = symmetric_factors(4, dim, T)
    return quote
        $(Expr(:meta, :inline))
        F = $F
        SV = tosimd(get_data(S))
        SVSV = SV * SV; FSVSV = F * SVSV; r = sum(FSVSV)
        return sqrt(r)
    end
end
