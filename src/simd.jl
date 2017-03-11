#=
This module contains explicit SIMD instructions for tensors.
Many of the methods defined outside this module will use SIMD-instructions
if julia is ran with -O3. Even if -O3 is enabled, the compiler is sometimes
thrown off guard, and therefore, explicit SIMD routines are
defined. This will enable SIMD-instructions even if julia is ran with
the default -O2.

The functions here are only defined for tensors of the same
element type. Otherwise it does work. Promotion should take
care of this before the tensors enter the functions here.

The module is organized as follows:
(1): + and - between tensors
(2): * and / between tensor and number
(3): dot
(4): dcontract
(5): otimes
(6): norm
=#
module ExplicitSIMD

using Tensors
using Tensors: AllTensors, get_data, n_components, get_base
using Compat

import SIMD
@compat const SVec{N, T} = SIMD.Vec{N, T}

const SIMDTypes = Union{Bool,
                        Int8, Int16, Int32, Int64, Int128,
                        UInt8, UInt16, UInt32, UInt64, UInt128,
                        Float16, Float32, Float64}

@compat const AllSIMDTensors{T <: SIMDTypes, dim} = AllTensors{dim, T} # T more useful so swapping

# SIMD sizes accepted by LLVM between 1 and 100
const SIMD_CHUNKS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 65, 66, 68, 72, 80, 96)

# factors for the symmetric tensors, return a quote
function symmetric_factors(order, dim, T)
    if order == 2
        dim == 1 && return :(SVec{1, T}((T(1),)))
        dim == 2 && return :(SVec{3, T}((T(1),T(2),T(1))))
        dim == 3 && return :(SVec{6, T}((T(1),T(2),T(2),T(1),T(2),T(1))))
    elseif order == 4
        dim == 1 && return :(SVec{1, T}((T(1),)))
        dim == 2 && return :(SVec{9, T}((T(1),T(2),T(1),T(2),T(4),T(2),T(1),T(2),T(1))))
        dim == 3 && return :(SVec{36,T}((T(1),T(2),T(2),T(1),T(2),T(1),T(2),T(4),T(4),T(2),T(4),T(2),
                                         T(2),T(4),T(4),T(2),T(4),T(2),T(1),T(2),T(2),T(1),T(2),T(1),
                                         T(2),T(4),T(4),T(2),T(4),T(2),T(1),T(2),T(2),T(1),T(2),T(1))))
    end
end

# Tensor from SVec (Note: This needs to be two separate methods in order to dispatch correctly)
@generated function (::Type{Tensor{order, dim}}){order, dim, N, T}(r::SVec{N, T})
    return quote
        $(Expr(:meta, :inline))
        Tensor{$order, $dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end
@generated function (::Type{SymmetricTensor{order, dim}}){order, dim, N, T}(r::SVec{N, T})
    return quote
        $(Expr(:meta, :inline))
        SymmetricTensor{$order, $dim}($(Expr(:tuple, [:(r[$i]) for i in 1:N]...)))
    end
end
# constructor from several SVecs which happens in dot, dcontract and otimes
@generated function (::Type{Tensor{order, dim}}){order, dim, M, N, T}(r::NTuple{M, SVec{N, T}})
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{$order, $dim}($(Expr(:tuple, [:(r[$j][$i]) for i in 1:N, j in 1:M]...)))
    end
end
@generated function (::Type{SymmetricTensor{order, dim}}){order, dim, M, N, T}(r::NTuple{M, SVec{N, T}})
    return quote
        $(Expr(:meta, :inline))
        @inbounds return SymmetricTensor{$order, $dim}($(Expr(:tuple, [:(r[$j][$i]) for i in 1:N, j in 1:M]...)))
    end
end

################################
# (1): + and - between tensors #
################################
@generated function Base.:+{TT <: AllSIMDTensors}(S1::TT, S2::TT)
    TensorType = get_base(S1)
    T = eltype(TT)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = SVec{$N, $T}(get_data(S1))
            D2 = SVec{$N, $T}(get_data(S2))
            r = D1 + D2
            return $TensorType(r)
        end
    end
end
@generated function Base.:+{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = get_data(S1)
            D2 = get_data(S2)
            D180 = SVec{80, T}($(Expr(:tuple, [:(D1[$i]) for i in 1:80]...)))
            D280 = SVec{80, T}($(Expr(:tuple, [:(D2[$i]) for i in 1:80]...)))
            r = D180 + D280
            r81 = D1[81] + D2[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:-{TT <: AllSIMDTensors}(S1::TT, S2::TT)
    TensorType = get_base(S1)
    T = eltype(TT)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = SVec{$N, $T}(get_data(S1))
            D2 = SVec{$N, $T}(get_data(S2))
            r = D1 - D2
            return $TensorType(r)
        end
    end
end
@generated function Base.:-{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D1 = get_data(S1)
            D2 = get_data(S2)
            D180 = SVec{80, T}($(Expr(:tuple, [:(D1[$i]) for i in 1:80]...)))
            D280 = SVec{80, T}($(Expr(:tuple, [:(D2[$i]) for i in 1:80]...)))
            r = D180 - D280
            r81 = D1[81] - D2[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end

##########################################
# (2): * and / between tensor and number #
##########################################
# note it is allowed with different eltypes, since it is promoted in SIMD.jl
@generated function Base.:*{T1 <: SIMDTypes, T2 <: SIMDTypes}(n::T1, S::AllSIMDTensors{T2})
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T2}(get_data(S))
            r = n * D
            return $TensorType(r)
        end
    end
end
@generated function Base.:*{T1 <: SIMDTypes, T2 <: SIMDTypes}(n::T1, S::Tensor{4, 3, T2})
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T2}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            r = n * D80
            r81 = n * D[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:*{T1 <: SIMDTypes, T2 <: SIMDTypes}(S::AllSIMDTensors{T1}, n::T2)
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T1}(get_data(S))
            r = D * n
            return $TensorType(r)
        end
    end
end
@generated function Base.:*{T1 <: SIMDTypes, T2 <: SIMDTypes}(S::Tensor{4, 3, T1}, n::T2)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T1}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            r = D80 * n
            r81 = D[81] * n
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:/{T1 <: SIMDTypes, T2 <: SIMDTypes}(S::AllSIMDTensors{T1}, n::T2)
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T1}(get_data(S))
            r = D / n
            return $TensorType(r)
        end
    end
end
@generated function Base.:/{T1 <: SIMDTypes, T2 <: SIMDTypes}(S::Tensor{4, 3, T1}, n::T2)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T1}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            r = D80 / n
            r81 = D[81] / n
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end

############
# (3): dot #
############
# 2-1
@inline function Base.dot{T <: SIMDTypes, N}(S1::Tensor{2, 2, T, N}, S2::Vec{2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{2, T}((D1[1], D1[2]))
        D12 = SVec{2, T}((D1[3], D1[4]))
        r = fma(D12, D2[2], D11 * D2[1])
        return Tensor{1, 2}(r)
    end
end
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 3, T}, S2::Vec{3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))
        D12 = SVec{3, T}((D1[4], D1[5], D1[6]))
        D13 = SVec{3, T}((D1[7], D1[8], D1[9]))
        r = fma(D13, D2[3], fma(D12, D2[2], D11 * D2[1]))
        return Tensor{1, 3}(r)
    end
end

# 2-2
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{2, T}((D1[1], D1[2]))
        D12 = SVec{2, T}((D1[3], D1[4]))
        r1 = fma(D12, D2[2], D11 * D2[1])
        r2 = fma(D12, D2[4], D11 * D2[3])
        return Tensor{2, 2}((r1, r2))
    end
end
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))
        D12 = SVec{3, T}((D1[4], D1[5], D1[6]))
        D13 = SVec{3, T}((D1[7], D1[8], D1[9]))
        r1 = fma(D13, D2[3], fma(D12, D2[2], D11 * D2[1]))
        r2 = fma(D13, D2[6], fma(D12, D2[5], D11 * D2[4]))
        r3 = fma(D13, D2[9], fma(D12, D2[8], D11 * D2[7]))
        return Tensor{2, 3}((r1, r2, r3))
    end
end

##################
# (4): dcontract #
##################
# 2-2
@inline function Tensors.dcontract{dim, T <: SIMDTypes, N}(S1::Tensor{2, dim, T, N}, S2::Tensor{2, dim, T, N})
    D1 = SVec{N, T}(get_data(S1))
    D2 = SVec{N, T}(get_data(S2))
    D1D2 = D1 * D2
    return sum(D1D2)
end
# 2s-2s
@generated function Tensors.dcontract{dim, T <: SIMDTypes, N}(S1::SymmetricTensor{2, dim, T, N}, S2::SymmetricTensor{2, dim, T, N})
    F = symmetric_factors(2, dim, T)
    return quote
        $(Expr(:meta, :inline))
        F = $F
        D1 = SVec{N, T}(get_data(S1))
        D2 = SVec{N, T}(get_data(S2))
        D1D2 = D1 * D2; FD1D2 = F * D1D2
        return sum(FD1D2)
    end
end

# 4-2
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))
        r = fma(D14, D2[4], fma(D13, D2[3], fma(D12, D2[2], D11 * D2[1])))
        return Tensor{2, 2}(r)
    end
end
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{9, T}((D1[1],  D1[2],  D1[3],  D1[4],  D1[5],  D1[6],  D1[7],  D1[8],  D1[9]))
        D12 = SVec{9, T}((D1[10], D1[11], D1[12], D1[13], D1[14], D1[15], D1[16], D1[17], D1[18]))
        D13 = SVec{9, T}((D1[19], D1[20], D1[21], D1[22], D1[23], D1[24], D1[25], D1[26], D1[27]))
        D14 = SVec{9, T}((D1[28], D1[29], D1[30], D1[31], D1[32], D1[33], D1[34], D1[35], D1[36]))
        D15 = SVec{9, T}((D1[37], D1[38], D1[39], D1[40], D1[41], D1[42], D1[43], D1[44], D1[45]))
        D16 = SVec{9, T}((D1[46], D1[47], D1[48], D1[49], D1[50], D1[51], D1[52], D1[53], D1[54]))
        D17 = SVec{9, T}((D1[55], D1[56], D1[57], D1[58], D1[59], D1[60], D1[61], D1[62], D1[63]))
        D18 = SVec{9, T}((D1[64], D1[65], D1[66], D1[67], D1[68], D1[69], D1[70], D1[71], D1[72]))
        D19 = SVec{9, T}((D1[73], D1[74], D1[75], D1[76], D1[77], D1[78], D1[79], D1[80], D1[81]))
        r = fma(D19, D2[9],  fma(D18, D2[8],  fma(D17, D2[7],  fma(D16, D2[6],  fma(D15, D2[5],  fma(D14, D2[4],  fma(D13, D2[3],  fma(D12, D2[2],  D11 * D2[1]))))))))
        return return Tensor{2, 3}(r)
    end
end

# 4-4
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{4, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))
        r1 = fma(D14, D2[4],  fma(D13, D2[3],  fma(D12, D2[2],  D11 * D2[1])))
        r2 = fma(D14, D2[8],  fma(D13, D2[7],  fma(D12, D2[6],  D11 * D2[5])))
        r3 = fma(D14, D2[12], fma(D13, D2[11], fma(D12, D2[10], D11 * D2[9])))
        r4 = fma(D14, D2[16], fma(D13, D2[15], fma(D12, D2[14], D11 * D2[13])))
        return Tensor{4, 2}((r1, r2, r3, r4))
    end
end
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{9, T}((D1[1],  D1[2],  D1[3],  D1[4],  D1[5],  D1[6],  D1[7],  D1[8],  D1[9]))
        D12 = SVec{9, T}((D1[10], D1[11], D1[12], D1[13], D1[14], D1[15], D1[16], D1[17], D1[18]))
        D13 = SVec{9, T}((D1[19], D1[20], D1[21], D1[22], D1[23], D1[24], D1[25], D1[26], D1[27]))
        D14 = SVec{9, T}((D1[28], D1[29], D1[30], D1[31], D1[32], D1[33], D1[34], D1[35], D1[36]))
        D15 = SVec{9, T}((D1[37], D1[38], D1[39], D1[40], D1[41], D1[42], D1[43], D1[44], D1[45]))
        D16 = SVec{9, T}((D1[46], D1[47], D1[48], D1[49], D1[50], D1[51], D1[52], D1[53], D1[54]))
        D17 = SVec{9, T}((D1[55], D1[56], D1[57], D1[58], D1[59], D1[60], D1[61], D1[62], D1[63]))
        D18 = SVec{9, T}((D1[64], D1[65], D1[66], D1[67], D1[68], D1[69], D1[70], D1[71], D1[72]))
        D19 = SVec{9, T}((D1[73], D1[74], D1[75], D1[76], D1[77], D1[78], D1[79], D1[80], D1[81]))
        r1 = fma(D19, D2[9],  fma(D18, D2[8],  fma(D17, D2[7],  fma(D16, D2[6],  fma(D15, D2[5],  fma(D14, D2[4],  fma(D13, D2[3],  fma(D12, D2[2],  D11 * D2[1] ))))))))
        r2 = fma(D19, D2[18], fma(D18, D2[17], fma(D17, D2[16], fma(D16, D2[15], fma(D15, D2[14], fma(D14, D2[13], fma(D13, D2[12], fma(D12, D2[11], D11 * D2[10]))))))))
        r3 = fma(D19, D2[27], fma(D18, D2[26], fma(D17, D2[25], fma(D16, D2[24], fma(D15, D2[23], fma(D14, D2[22], fma(D13, D2[21], fma(D12, D2[20], D11 * D2[19]))))))))
        r4 = fma(D19, D2[36], fma(D18, D2[35], fma(D17, D2[34], fma(D16, D2[33], fma(D15, D2[32], fma(D14, D2[31], fma(D13, D2[30], fma(D12, D2[29], D11 * D2[28]))))))))
        r5 = fma(D19, D2[45], fma(D18, D2[44], fma(D17, D2[43], fma(D16, D2[42], fma(D15, D2[41], fma(D14, D2[40], fma(D13, D2[39], fma(D12, D2[38], D11 * D2[37]))))))))
        r6 = fma(D19, D2[54], fma(D18, D2[53], fma(D17, D2[52], fma(D16, D2[51], fma(D15, D2[50], fma(D14, D2[49], fma(D13, D2[48], fma(D12, D2[47], D11 * D2[46]))))))))
        r7 = fma(D19, D2[63], fma(D18, D2[62], fma(D17, D2[61], fma(D16, D2[60], fma(D15, D2[59], fma(D14, D2[58], fma(D13, D2[57], fma(D12, D2[56], D11 * D2[55]))))))))
        r8 = fma(D19, D2[72], fma(D18, D2[71], fma(D17, D2[70], fma(D16, D2[69], fma(D15, D2[68], fma(D14, D2[67], fma(D13, D2[66], fma(D12, D2[65], D11 * D2[64]))))))))
        r9 = fma(D19, D2[81], fma(D18, D2[80], fma(D17, D2[79], fma(D16, D2[78], fma(D15, D2[77], fma(D14, D2[76], fma(D13, D2[75], fma(D12, D2[74], D11 * D2[73]))))))))
        return Tensor{4, 3}((r1, r2, r3, r4, r5, r6, r7, r8, r9))
    end
end

###############
# (5): otimes #
###############
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Vec{2, T}, S2::Vec{2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{2, T}((D1[1], D1[2]))
        r1 = D11 * D2[1]
        r2 = D11 * D2[2]
        return Tensor{2, 2}((r1, r2))
    end
end
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Vec{3, T}, S2::Vec{3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))
        r1 = D11 * D2[1]; r2 = D11 * D2[2]; r3 = D11 * D2[3]
        return Tensor{2, 3}((r1, r2, r3))
    end
end
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}(D1)
        r1 = D11 * D2[1]; r2 = D11 * D2[2]; r3 = D11 * D2[3]; r4 = D11 * D2[4]
        return Tensor{4, 2}((r1, r2, r3, r4))
    end
end
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{9, T}(D1)
        r1 = D11 * D2[1]; r2 = D11 * D2[2]; r3 = D11 * D2[3]
        r4 = D11 * D2[4]; r5 = D11 * D2[5]; r6 = D11 * D2[6]
        r7 = D11 * D2[7]; r8 = D11 * D2[8]; r9 = D11 * D2[9]
        return Tensor{4, 3}((r1, r2, r3, r4, r5, r6, r7, r8, r9))
    end
end
@inline function Tensors.otimes{T <: SIMDTypes}(S1::SymmetricTensor{2, 2, T}, S2::SymmetricTensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}(D1)
        r1 = D11 * D2[1]; r2 = D11 * D2[2]; r3 = D11 * D2[3]
        return SymmetricTensor{4, 2}((r1, r2, r3))
    end
end
@inline function Tensors.otimes{T <: SIMDTypes}(S1::SymmetricTensor{2, 3, T}, S2::SymmetricTensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{6, T}(D1)
        r1 = D11 * D2[1]; r2 = D11 * D2[2]; r3 = D11 * D2[3]
        r4 = D11 * D2[4]; r5 = D11 * D2[5]; r6 = D11 * D2[6]
        return SymmetricTensor{4, 3}((r1, r2, r3, r4, r5, r6))
    end
end

#############
# (6): norm #
#############
# order 1 and order 2 norms rely on dot and dcontract respectively
@inline function Base.norm{T <: SIMDTypes, N}(S::Tensor{4, 2, T, N})
    @inbounds begin
        D = SVec{N, T}(get_data(S))
        DD = D * D
        r = sum(DD)
        return sqrt(r)
    end
end
@generated function Base.norm{T <: SIMDTypes}(S::Tensor{4, 3, T})
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            D80D80 = D80 * D80
            r80 = sum(D80D80)
            r = r80 + D[81] * D[81]
            return sqrt(r)
        end
    end
end
@generated function Base.norm{dim, T <: SIMDTypes, N}(S::SymmetricTensor{4, dim, T, N})
    F = symmetric_factors(4, dim, T)
    return quote
        $(Expr(:meta, :inline))
        F = $F
        D = SVec{N, T}(get_data(S))
        DD = D * D; FDD = F * DD; r = sum(FDD)
        return sqrt(r)
    end
end

end # module
