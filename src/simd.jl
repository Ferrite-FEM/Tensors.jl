#=
This module contains explicit SIMD instructions for tensors.
Many of the methods defined outside this module will use SIMD-instructions
if julia is ran with -O3. Even if -O3 is enabled, the compiler is sometimes
thrown off guard, and therefore, explicit SIMD routines are
defined. This will enable SIMD-instructions even if julia is ran with
the default -O2.

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

@compat const AllSIMDTensors{dim, T <: SIMDTypes} = AllTensors{dim, T}

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
@generated function Base.:*{dim, T <: SIMDTypes}(n::T, S::AllTensors{dim, T})
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T}(get_data(S))
            r = n * D
            return $TensorType(r)
        end
    end
end
@generated function Base.:*{T <: SIMDTypes}(n::T, S::Tensor{4, 3, T})
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            r = n * D80
            r81 = n * D[81]
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:*{dim, T <: SIMDTypes}(S::AllTensors{dim, T}, n::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T}(get_data(S))
            r = D * n
            return $TensorType(r)
        end
    end
end
@generated function Base.:*{T <: SIMDTypes}(S::Tensor{4, 3, T}, n::T)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
            r = D80 * n
            r81 = D[81] * n
            return Tensor{4, 3}($(Expr(:tuple, [:(r[$i]) for i in 1:80]..., :(r81))))
        end
    end
end
@generated function Base.:/{dim, T <: SIMDTypes}(S::AllTensors{dim, T}, n::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = SVec{$N, T}(get_data(S))
            r = D / n
            return $TensorType(r)
        end
    end
end
@generated function Base.:/{T <: SIMDTypes}(S::Tensor{4, 3, T}, n::T)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            D = get_data(S)
            D80 = SVec{80, T}($(Expr(:tuple, [:(D[$i]) for i in 1:80]...)))
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
@inline function Base.dot{T <: SIMDTypes, N}(S1::Tensor{2, 1, T, N}, S2::Vec{1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1],))
        r = D11 * D2[1]
        return Tensor{1, 1}(r)
    end
end
@inline function Base.dot{T <: SIMDTypes, N}(S1::Tensor{2, 2, T, N}, S2::Vec{2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{2, T}((D1[1], D1[2]))
        D12 = SVec{2, T}((D1[3], D1[4]))
        r1 = D11 * D2[1]; r2 = D12 * D2[2]
        r = r1 + r2
        return Tensor{1, 2}(r)
    end
end
@inline function Base.dot{T <: SIMDTypes, N}(S1::Tensor{2, 3, T, N}, S2::Vec{3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))
        D12 = SVec{3, T}((D1[4], D1[5], D1[6]))
        D13 = SVec{3, T}((D1[7], D1[8], D1[9]))
        r1 = D11 * D2[1]; r2 = D12 * D2[2]; r3 = D13 * D2[3]
        r12 = r1 + r2; r = r12 + r3
        return Tensor{1, 3}(r)
    end
end

# 2-2
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 1, T}, S2::Tensor{2, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1], ))
        r1 = D11 * D2[1]
        return Tensor{2, 1}((r1,))
    end
end
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{2, T}((D1[1], D1[2]))
        D12 = SVec{2, T}((D1[3], D1[4]))
        r1 = D11 * D2[1]; r2 = D12 * D2[2]; r12 = r1 + r2
        r3 = D11 * D2[3]; r4 = D12 * D2[4]; r34 = r3 + r4
        return Tensor{2, 2}((r12, r34))
    end
end
@inline function Base.dot{T <: SIMDTypes}(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))
        D12 = SVec{3, T}((D1[4], D1[5], D1[6]))
        D13 = SVec{3, T}((D1[7], D1[8], D1[9]))
        r1 = D11 * D2[1]; r2 = D12 * D2[2]; r3 = D13 * D2[3]
        r12 = r1 + r2; r123 = r12 + r3
        r4 = D11 * D2[4]; r5 = D12 * D2[5]; r6 = D13 * D2[6]
        r45 = r4 + r5; r456 = r45 + r6
        r7 = D11 * D2[7]; r8 = D12 * D2[8]; r9 = D13 * D2[9]
        r78 = r7 + r8; r789 = r78 + r9
        return Tensor{2, 3}((r123, r456, r789))
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
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 1, T}, S2::Tensor{2, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1], ))
        r  = D11 * D2[1]
        return Tensor{2, 1}(r)
    end
end
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))
        r1 = D11 * D2[1]; r2 = D12 * D2[2]
        r3 = D13 * D2[3]; r4 = D14 * D2[4]
        r12 = r1 + r2; r34 = r3 + r4; r = r12 + r34
        return Tensor{2, 2}(r)
    end
end
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4,3,T}, S2::Tensor{2,3,T})
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
        r1 = D11 * D2[1]; r2 = D12 * D2[2]; r3 = D13 * D2[3]
        r4 = D14 * D2[4]; r5 = D15 * D2[5]; r6 = D16 * D2[6]
        r7 = D17 * D2[7]; r8 = D18 * D2[8]; r9 = D19 * D2[9]
        r12 = r1 + r2; r34 = r3 + r4; r56 = r5 + r6; r78 = r7 + r8
        r1234 = r12 + r34; r5678 = r56 + r78; r12345678 = r1234 + r5678
        r = r12345678 + r9
        return Tensor{2, 3}(r)
    end
end

# 4-4
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 1, T}, S2::Tensor{4, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1], ))
        r1  = D11 * D2[1]
        return Tensor{4, 1}((r1, ))
    end
end
@inline function Tensors.dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{4, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))
        r11 = D11 * D2[1]; r12 = D12 * D2[2]; r13 = D13 * D2[3]; r14 = D14 * D2[4]
        r112 = r11 + r12; r134 = r13 + r14; r1 = r112 + r134
        r21 = D11 * D2[5]; r22 = D12 * D2[6]; r23 = D13 * D2[7]; r24 = D14 * D2[8]
        r212 = r21 + r22; r234 = r23 + r24; r2 = r212 + r234
        r31 = D11 * D2[9]; r32 = D12 * D2[10]; r33 = D13 * D2[11]; r34 = D14 * D2[12]
        r312 = r31 + r32; r334 = r33 + r34; r3 = r312 + r334
        r41 = D11 * D2[13]; r42 = D12 * D2[14]; r43 = D13 * D2[15]; r44 = D14 * D2[16]
        r412 = r41 + r42; r434 = r43 + r44; r4 = r412 + r434
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
        r11 = D11 * D2[1]; r12 = D12 * D2[2]; r13 = D13 * D2[3]
        r14 = D14 * D2[4]; r15 = D15 * D2[5]; r16 = D16 * D2[6]
        r17 = D17 * D2[7]; r18 = D18 * D2[8]; r19 = D19 * D2[9]
        r112 = r11 + r12; r134 = r13 + r14; r156 = r15 + r16; r178 = r17 + r18
        r11234 = r112 + r134; r15678 = r156 + r178; r112345678 = r11234 + r15678
        r1 = r112345678 + r19
        r21 = D11 * D2[10]; r22 = D12 * D2[11]; r23 = D13 * D2[12]
        r24 = D14 * D2[13]; r25 = D15 * D2[14]; r26 = D16 * D2[15]
        r27 = D17 * D2[16]; r28 = D18 * D2[17]; r29 = D19 * D2[18]
        r212 = r21 + r22; r234 = r23 + r24; r256 = r25 + r26; r278 = r27 + r28
        r21234 = r212 + r234; r25678 = r256 + r278; r212345678 = r21234 + r25678
        r2 = r212345678 + r29
        r31 = D11 * D2[19]; r32 = D12 * D2[20]; r33 = D13 * D2[21]
        r34 = D14 * D2[22]; r35 = D15 * D2[23]; r36 = D16 * D2[24]
        r37 = D17 * D2[25]; r38 = D18 * D2[26]; r39 = D19 * D2[27]
        r312 = r31 + r32; r334 = r33 + r34; r356 = r35 + r36; r378 = r37 + r38
        r31234 = r312 + r334; r35678 = r356 + r378; r312345678 = r31234 + r35678
        r3 = r312345678 + r39
        r41 = D11 * D2[28]; r42 = D12 * D2[29]; r43 = D13 * D2[30]
        r44 = D14 * D2[31]; r45 = D15 * D2[32]; r46 = D16 * D2[33]
        r47 = D17 * D2[34]; r48 = D18 * D2[35]; r49 = D19 * D2[36]
        r412 = r41 + r42; r434 = r43 + r44; r456 = r45 + r46; r478 = r47 + r48
        r41234 = r412 + r434; r45678 = r456 + r478; r412345678 = r41234 + r45678
        r4 = r412345678 + r49
        r51 = D11 * D2[37]; r52 = D12 * D2[38]; r53 = D13 * D2[39]
        r54 = D14 * D2[40]; r55 = D15 * D2[41]; r56 = D16 * D2[42]
        r57 = D17 * D2[43]; r58 = D18 * D2[44]; r59 = D19 * D2[45]
        r512 = r51 + r52; r534 = r53 + r54; r556 = r55 + r56; r578 = r57 + r58
        r51234 = r512 + r534; r55678 = r556 + r578; r512345678 = r51234 + r55678
        r5 = r512345678 + r59
        r61 = D11 * D2[46]; r62 = D12 * D2[47]; r63 = D13 * D2[48]
        r64 = D14 * D2[49]; r65 = D15 * D2[50]; r66 = D16 * D2[51]
        r67 = D17 * D2[52]; r68 = D18 * D2[53]; r69 = D19 * D2[54]
        r612 = r61 + r62; r634 = r63 + r64; r656 = r65 + r66; r678 = r67 + r68
        r61234 = r612 + r634; r65678 = r656 + r678; r612345678 = r61234 + r65678
        r6 = r612345678 + r69
        r71 = D11 * D2[55]; r72 = D12 * D2[56]; r73 = D13 * D2[57]
        r74 = D14 * D2[58]; r75 = D15 * D2[59]; r76 = D16 * D2[60]
        r77 = D17 * D2[61]; r78 = D18 * D2[62]; r79 = D19 * D2[63]
        r712 = r71 + r72; r734 = r73 + r74; r756 = r75 + r76; r778 = r77 + r78
        r71234 = r712 + r734; r75678 = r756 + r778; r712345678 = r71234 + r75678
        r7 = r712345678 + r79
        r81 = D11 * D2[64]; r82 = D12 * D2[65]; r83 = D13 * D2[66]
        r84 = D14 * D2[67]; r85 = D15 * D2[68]; r86 = D16 * D2[69]
        r87 = D17 * D2[70]; r88 = D18 * D2[71]; r89 = D19 * D2[72]
        r812 = r81 + r82; r834 = r83 + r84; r856 = r85 + r86; r878 = r87 + r88
        r81234 = r812 + r834; r85678 = r856 + r878; r812345678 = r81234 + r85678
        r8 = r812345678 + r89
        r91 = D11 * D2[73]; r92 = D12 * D2[74]; r93 = D13 * D2[75]
        r94 = D14 * D2[76]; r95 = D15 * D2[77]; r96 = D16 * D2[78]
        r97 = D17 * D2[79]; r98 = D18 * D2[80]; r99 = D19 * D2[81]
        r912 = r91 + r92; r934 = r93 + r94; r956 = r95 + r96; r978 = r97 + r98
        r91234 = r912 + r934; r95678 = r956 + r978; r912345678 = r91234 + r95678
        r9 = r912345678 + r99
        return Tensor{4, 3}((r1, r2, r3, r4, r5, r6, r7, r8, r9))
    end
end

###############
# (5): otimes #
###############
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Vec{1, T}, S2::Vec{1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1], ))
        r1 = D11 * D2[1]
        return Tensor{2, 1}((r1, ))
    end
end
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
@inline function Tensors.otimes{T <: SIMDTypes}(S1::Tensor{2, 1, T}, S2::Tensor{2, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}(D1)
        r1 = D11 * D2[1]
        return Tensor{4, 1}((r1[1], ))
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
@inline function Base.norm{dim, T <: SIMDTypes, N}(S::Tensor{4, dim, T, N})
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
