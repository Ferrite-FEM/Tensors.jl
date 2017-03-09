# this files contains some performance related stuff
module ST # SIMDTensors

using Tensors
using Tensors: get_data, AllTensors, n_components, get_base
using Compat

import SIMD
@compat const SVec{N, T} = SIMD.Vec{N, T}

const SIMDTypes = Union{Bool,
                        Int8, Int16, Int32, Int32, Int128,
                        UInt8, UInt16, UInt32, UInt32, UInt128,
                        Float16, Float32, Float64}

@compat const AllSIMDTensors{dim, T <: SIMDTypes} = AllTensors{dim, T}

# SIMD sizes accepted by LLVM between 1 and 100
const SIMD_CHUNKS = (1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 16, 17, 18, 20, 24, 32, 33, 34, 36, 40, 48, 64, 65, 66, 68, 72, 80, 96)

# factors for the symmetric tensors
#=Base.@pure=# function symmetric_factors(order, dim, T) # can we have @pure here? Then we don't need as many generated functions
    if order == 2
        dim == 1 && return SVec{1, T}((T(1),))
        dim == 2 && return SVec{3, T}((T(1),T(2),T(1)))
        dim == 3 && return SVec{6, T}((T(1),T(2),T(2),T(1),T(2),T(1)))
    elseif order == 4
        dim == 1 && return SVec{1, T}((T(1),))
        dim == 2 && return SVec{9, T}((T(1),T(2),T(1),T(2),T(4),T(2),T(1),T(2),T(1)))
        dim == 3 && return SVec{36,T}((T(1),T(2),T(2),T(1),T(2),T(1),T(2),T(4),T(4),T(2),T(4),T(2),
                                       T(2),T(4),T(4),T(2),T(4),T(2),T(1),T(2),T(2),T(1),T(2),T(1),
                                       T(2),T(4),T(4),T(2),T(4),T(2),T(1),T(2),T(2),T(1),T(2),T(1)))
    end
end

# norm
@inline function norm{order, dim, T <: SIMDTypes, N}(S::Tensor{order, dim, T, N})
    D = SVec{N, T}(get_data(S))
    DD = D * D
    sDD = sum(DD)
    return sqrt(sDD)
end
@inline function norm{T <: SIMDTypes}(S::Tensor{4, 3, T})
    @inbounds begin
        D = get_data(S)
        D80 = SVec{80, T}((D[1],  D[2],  D[3],  D[4],  D[5],  D[6],  D[7],  D[8],  D[9],
                           D[10], D[11], D[12], D[13], D[14], D[15], D[16], D[17], D[18],
                           D[19], D[20], D[21], D[22], D[23], D[24], D[25], D[26], D[27],
                           D[28], D[29], D[30], D[31], D[32], D[33], D[34], D[35], D[36],
                           D[37], D[38], D[39], D[40], D[41], D[42], D[43], D[44], D[45],
                           D[46], D[47], D[48], D[49], D[50], D[51], D[52], D[53], D[54],
                           D[55], D[56], D[57], D[58], D[59], D[60], D[61], D[62], D[63],
                           D[64], D[65], D[66], D[67], D[68], D[69], D[70], D[71], D[72],
                           D[73], D[74], D[75], D[76], D[77], D[78], D[79], D[80]))
        D80D80 = D80 * D80
        v = sum(D80D80)
        v += D[81] * D[81]
        return sqrt(v)
    end
end

# rely on fast dcontract (this might be slower actually since dcontract assumes two different tensors)
# Edit: no, the computer understands its the same #sosmart
# @inline norm{dim, T <: SIMDTypes}(S::SymmetricTensor{2, dim, T}) = sqrt(dcontract(S, S))

@generated function norm{order, dim, T <: SIMDTypes, N}(S::SymmetricTensor{order, dim, T, N})
    F = symmetric_factors(order, dim, T)
    return quote
        $(Expr(:meta, :inline))
        F = $F
        D = SVec{N, T}(get_data(S))
        DD = D * D; FDD = F * DD; sFDD = sum(FDD)
        return sqrt(sFDD)
    end
end

# This is the same if it is safe to have @pure on symmetric_factors
# But like this we dont use a generated function.
# @inline function norm{order, dim, T <: SIMDTypes, N}(S::SymmetricTensor{order, dim, T, N})
#     F = symmetric_factors(order, dim, T)
#     D = SVec{N, T}(get_data(S))
#     DD = D * D; FDD = F * DD; sFDD = sum(FDD)
#     return sqrt(sFDD)
# end

# dcontract
@inline dcontract{dim, T <: SIMDTypes}(S1::SecondOrderTensor{dim, T}, S2::SecondOrderTensor{dim, T}) = dcontract(promote(S1, S2)...)

@inline function dcontract{dim, T <: SIMDTypes, N}(S1::Tensor{2, dim, T, N}, S2::Tensor{2, dim, T, N})
    D1 = SVec{N, T}(get_data(S1))
    D2 = SVec{N, T}(get_data(S2))
    D1D2 = D1 * D2
    return sum(D1D2)
end

# this can also be a non-generated function if we can have @pure
@generated function dcontract{T <: SIMDTypes}(S1::SymmetricTensor{2, dim, T, N}, S2::SymmetricTensor{2, dim, T, N})
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

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4, 1, T}, S2::Tensor{2, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{4, T}((D1[1], ))
        r  = D11 * D2[1]
        return Tensor{2, 1}((r[1], ))
    end
end

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))

        r  = D11 * D2[1]
        r += D12 * D2[2]
        r += D13 * D2[3]
        r += D14 * D2[4]

        return Tensor{2, 2}((r[1], r[2], r[3], r[4]))
    end
end

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4,3,T}, S2::Tensor{2,3,T})
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

        r  = D11 * D2[1]
        r += D12 * D2[2]
        r += D13 * D2[3]
        r += D14 * D2[4]
        r += D15 * D2[5]
        r += D16 * D2[6]
        r += D17 * D2[7]
        r += D18 * D2[8]
        r += D19 * D2[9]

        return Tensor{2, 3}((r[1], r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9]))
    end
end

# @inline dcontract2{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{4, 2, T}) = dcontract(majortranspose(S2), S1)

# @inline function dcontract4{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{4, 2, T})
#     @inbounds begin
        # D1 = get_data(S1)
        # D2 = get_data(S2)

#         D1 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
#         D21 = SVec{4, T}((D2[1],  D2[2],  D2[3],  D2[4]))
#         D22 = SVec{4, T}((D2[5],  D2[6],  D2[7],  D2[8]))
#         D23 = SVec{4, T}((D2[9],  D2[10], D2[11], D2[12]))
#         D24 = SVec{4, T}((D2[13], D2[14], D2[15], D2[16]))

#         D1D21 = D1 * D21
#         r1  = sum(D1D21)
#         D1D22 = D1 * D22
#         r2  = sum(D1D22)
#         D1D23 = D1 * D23
#         r3  = sum(D1D23)
#         D1D24 = D1 * D24
#         r4  = sum(D1D24)

#         return Tensor{2, 2}((r1, r2, r3, r4))
#     end
# end

# @inline function dcontract3{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{4, 2, T})
#     @inbounds begin
        # D1 = get_data(S1)
        # D2 = get_data(S2)

#         D21 = SVec{4, T}((D2[1], D2[5], D2[9],  D2[13]))
#         D22 = SVec{4, T}((D2[2], D2[6], D2[10], D2[14]))
#         D23 = SVec{4, T}((D2[3], D2[7], D2[11], D2[15]))
#         D24 = SVec{4, T}((D2[4], D2[8], D2[12], D2[16]))

#         r  = D21 * D1[1]
#         r += D22 * D1[2]
#         r += D23 * D1[3]
#         r += D24 * D1[4]

#         return Tensor{2, 2}((r[1], r[2], r[3], r[4]))
#     end
# end

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4, 1, T}, S2::Tensor{4, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D11 = SVec{1, T}((D1[1], ))
        r1  = D11 * D2[1]
        return Tensor{4, 1}((r1[1], ))
    end
end

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4, 2, T}, S2::Tensor{4, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{4, T}((D1[1],  D1[2],  D1[3],  D1[4]))
        D12 = SVec{4, T}((D1[5],  D1[6],  D1[7],  D1[8]))
        D13 = SVec{4, T}((D1[9],  D1[10], D1[11], D1[12]))
        D14 = SVec{4, T}((D1[13], D1[14], D1[15], D1[16]))

        r1  = D11 * D2[1]
        r1 += D12 * D2[2]
        r1 += D13 * D2[3]
        r1 += D14 * D2[4]

        r2  = D11 * D2[5]
        r2 += D12 * D2[6]
        r2 += D13 * D2[7]
        r2 += D14 * D2[8]

        r3  = D11 * D2[9]
        r3 += D12 * D2[10]
        r3 += D13 * D2[11]
        r3 += D14 * D2[12]

        r4  = D11 * D2[13]
        r4 += D12 * D2[14]
        r4 += D13 * D2[15]
        r4 += D14 * D2[16]

        return Tensor{4, 2}((r1[1],  r1[2],  r1[3],  r1[4],
                             r2[1],  r2[2],  r2[3],  r2[4],
                             r3[1],  r3[2],  r3[3],  r3[4],
                             r4[1],  r4[2],  r4[3],  r4[4]))
    end
end

@inline function dcontract{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
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

        r1  = D11 * D2[1]
        r1 += D12 * D2[2]
        r1 += D13 * D2[3]
        r1 += D14 * D2[4]
        r1 += D15 * D2[5]
        r1 += D16 * D2[6]
        r1 += D17 * D2[7]
        r1 += D18 * D2[8]
        r1 += D19 * D2[9]

        r2  = D11 * D2[10]
        r2 += D12 * D2[11]
        r2 += D13 * D2[12]
        r2 += D14 * D2[13]
        r2 += D15 * D2[14]
        r2 += D16 * D2[15]
        r2 += D17 * D2[16]
        r2 += D18 * D2[17]
        r2 += D19 * D2[18]

        r3  = D11 * D2[19]
        r3 += D12 * D2[20]
        r3 += D13 * D2[21]
        r3 += D14 * D2[22]
        r3 += D15 * D2[23]
        r3 += D16 * D2[24]
        r3 += D17 * D2[25]
        r3 += D18 * D2[26]
        r3 += D19 * D2[27]

        r4  = D11 * D2[28]
        r4 += D12 * D2[29]
        r4 += D13 * D2[30]
        r4 += D14 * D2[31]
        r4 += D15 * D2[32]
        r4 += D16 * D2[33]
        r4 += D17 * D2[34]
        r4 += D18 * D2[35]
        r4 += D19 * D2[36]

        r5  = D11 * D2[37]
        r5 += D12 * D2[38]
        r5 += D13 * D2[39]
        r5 += D14 * D2[40]
        r5 += D15 * D2[41]
        r5 += D16 * D2[42]
        r5 += D17 * D2[43]
        r5 += D18 * D2[44]
        r5 += D19 * D2[45]

        r6  = D11 * D2[46]
        r6 += D12 * D2[47]
        r6 += D13 * D2[48]
        r6 += D14 * D2[49]
        r6 += D15 * D2[50]
        r6 += D16 * D2[51]
        r6 += D17 * D2[52]
        r6 += D18 * D2[53]
        r6 += D19 * D2[54]

        r7  = D11 * D2[55]
        r7 += D12 * D2[56]
        r7 += D13 * D2[57]
        r7 += D14 * D2[58]
        r7 += D15 * D2[59]
        r7 += D16 * D2[60]
        r7 += D17 * D2[61]
        r7 += D18 * D2[62]
        r7 += D19 * D2[63]

        r8  = D11 * D2[64]
        r8 += D12 * D2[65]
        r8 += D13 * D2[66]
        r8 += D14 * D2[67]
        r8 += D15 * D2[68]
        r8 += D16 * D2[69]
        r8 += D17 * D2[70]
        r8 += D18 * D2[71]
        r8 += D19 * D2[72]

        r9  = D11 * D2[73]
        r9 += D12 * D2[74]
        r9 += D13 * D2[75]
        r9 += D14 * D2[76]
        r9 += D15 * D2[77]
        r9 += D16 * D2[78]
        r9 += D17 * D2[79]
        r9 += D18 * D2[80]
        r9 += D19 * D2[81]

        return Tensor{4, 3}((r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8], r1[9],
                             r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], r2[7], r2[8], r2[9],
                             r3[1], r3[2], r3[3], r3[4], r3[5], r3[6], r3[7], r3[8], r3[9],
                             r4[1], r4[2], r4[3], r4[4], r4[5], r4[6], r4[7], r4[8], r4[9],
                             r5[1], r5[2], r5[3], r5[4], r5[5], r5[6], r5[7], r5[8], r5[9],
                             r6[1], r6[2], r6[3], r6[4], r6[5], r6[6], r6[7], r6[8], r6[9],
                             r7[1], r7[2], r7[3], r7[4], r7[5], r7[6], r7[7], r7[8], r7[9],
                             r8[1], r8[2], r8[3], r8[4], r8[5], r8[6], r8[7], r8[8], r8[9],
                             r9[1], r9[2], r9[3], r9[4], r9[5], r9[6], r9[7], r9[8], r9[9]))
    end
end

# otimes
@inline function otimes{T <: SIMDTypes}(S1::Vec{1, T}, S2::Vec{1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{1, T}((D1[1], ))

        r1 = D11 * D2[1]

        return Tensor{2, 1}((r1[1], ))
    end
end

@inline function otimes{T <: SIMDTypes}(S1::Vec{2, T}, S2::Vec{2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{2, T}((D1[1], D1[2]))

        r1 = D11 * D2[1]
        r2 = D11 * D2[2]

        return Tensor{2, 2}((r1[1], r1[2], r2[1], r2[2]))
    end
end

@inline function otimes{T <: SIMDTypes}(S1::Vec{3, T}, S2::Vec{3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{3, T}((D1[1], D1[2], D1[3]))

        r1 = D11 * D2[1]
        r2 = D11 * D2[2]
        r3 = D11 * D2[3]

        return Tensor{2, 3}((r1[1], r1[2], r1[3],
                             r2[1], r2[2], r2[3],
                             r3[1], r3[2], r3[3]))
    end
end

@inline function otimes{T <: SIMDTypes}(S1::Tensor{2, 1, T}, S2::Tensor{2, 1, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{1, T}((D1[1], ))

        r1 = D11 * D2[1]

        return Tensor{4, 1}((r1[1], ))
    end
end

@inline function otimes{T <: SIMDTypes}(S1::Tensor{2, 2, T}, S2::Tensor{2, 2, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{4, T}((D1[1], D1[2], D1[3], D1[4]))

        r1 = D11 * D2[1]
        r2 = D11 * D2[2]
        r3 = D11 * D2[3]
        r4 = D11 * D2[4]

        return Tensor{4, 2}((r1[1], r1[2], r1[3], r1[4],
                             r2[1], r2[2], r2[3], r2[4],
                             r3[1], r3[2], r3[3], r3[4],
                             r4[1], r4[2], r4[3], r4[4]))
    end
end

@inline function otimes{T <: SIMDTypes}(S1::Tensor{2, 3, T}, S2::Tensor{2, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)

        D11 = SVec{9, T}((D1[1], D1[2], D1[3], D1[4], D1[5], D1[6], D1[7], D1[8], D1[9]))

        r1 = D11 * D2[1]
        r2 = D11 * D2[2]
        r3 = D11 * D2[3]
        r4 = D11 * D2[4]
        r5 = D11 * D2[5]
        r6 = D11 * D2[6]
        r7 = D11 * D2[7]
        r8 = D11 * D2[8]
        r9 = D11 * D2[9]

        return Tensor{4, 3}((r1[1], r1[2], r1[3], r1[4], r1[5], r1[6], r1[7], r1[8], r1[9],
                             r2[1], r2[2], r2[3], r2[4], r2[5], r2[6], r2[7], r2[8], r2[9],
                             r3[1], r3[2], r3[3], r3[4], r3[5], r3[6], r3[7], r3[8], r3[9],
                             r4[1], r4[2], r4[3], r4[4], r4[5], r4[6], r4[7], r4[8], r4[9],
                             r5[1], r5[2], r5[3], r5[4], r5[5], r5[6], r5[7], r5[8], r5[9],
                             r6[1], r6[2], r6[3], r6[4], r6[5], r6[6], r6[7], r6[8], r6[9],
                             r7[1], r7[2], r7[3], r7[4], r7[5], r7[6], r7[7], r7[8], r7[9],
                             r8[1], r8[2], r8[3], r8[4], r8[5], r8[6], r8[7], r8[8], r8[9],
                             r9[1], r9[2], r9[3], r9[4], r9[5], r9[6], r9[7], r9[8], r9[9]))
    end
end

# +
@generated function add{TT <: AllSIMDTensors}(S1::TT, S2::TT)
    TensorType = get_base(S1)
    T = eltype(TT)
    N = n_components(TensorType)
    D1 = :(D1 = SVec{$N, $T}(get_data(S1)))
    D2 = :(D2 = SVec{$N, $T}(get_data(S2)))
    r = :(r = D1 + D2)
    expr = Expr(:tuple, [:(r[$i]) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $D1
            $D2
            $r
            return $TensorType($expr)
        end
    end
end
@inline function add{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D180 = SVec{80, T}((D1[1],  D1[2],  D1[3],  D1[4],  D1[5],  D1[6],  D1[7],  D1[8],  D1[9],
                            D1[10], D1[11], D1[12], D1[13], D1[14], D1[15], D1[16], D1[17], D1[18],
                            D1[19], D1[20], D1[21], D1[22], D1[23], D1[24], D1[25], D1[26], D1[27],
                            D1[28], D1[29], D1[30], D1[31], D1[32], D1[33], D1[34], D1[35], D1[36],
                            D1[37], D1[38], D1[39], D1[40], D1[41], D1[42], D1[43], D1[44], D1[45],
                            D1[46], D1[47], D1[48], D1[49], D1[50], D1[51], D1[52], D1[53], D1[54],
                            D1[55], D1[56], D1[57], D1[58], D1[59], D1[60], D1[61], D1[62], D1[63],
                            D1[64], D1[65], D1[66], D1[67], D1[68], D1[69], D1[70], D1[71], D1[72],
                            D1[73], D1[74], D1[75], D1[76], D1[77], D1[78], D1[79], D1[80]))
        D280 = SVec{80, T}((D2[1],  D2[2],  D2[3],  D2[4],  D2[5],  D2[6],  D2[7],  D2[8],  D2[9],
                            D2[10], D2[11], D2[12], D2[13], D2[14], D2[15], D2[16], D2[17], D2[18],
                            D2[19], D2[20], D2[21], D2[22], D2[23], D2[24], D2[25], D2[26], D2[27],
                            D2[28], D2[29], D2[30], D2[31], D2[32], D2[33], D2[34], D2[35], D2[36],
                            D2[37], D2[38], D2[39], D2[40], D2[41], D2[42], D2[43], D2[44], D2[45],
                            D2[46], D2[47], D2[48], D2[49], D2[50], D2[51], D2[52], D2[53], D2[54],
                            D2[55], D2[56], D2[57], D2[58], D2[59], D2[60], D2[61], D2[62], D2[63],
                            D2[64], D2[65], D2[66], D2[67], D2[68], D2[69], D2[70], D2[71], D2[72],
                            D2[73], D2[74], D2[75], D2[76], D2[77], D2[78], D2[79], D2[80]))
        r = D180 + D280
        r81 = D1[81] + D2[81]
        return Tensor{4, 3}((r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                             r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                             r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                             r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                             r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                             r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                             r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                             r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                             r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r81))
    end
end

# -
@generated function sub{TT <: AllSIMDTensors}(S1::TT, S2::TT)
    TensorType = get_base(S1)
    T = eltype(TT)
    N = n_components(TensorType)
    D1 = :(D1 = SVec{$N, $T}(get_data(S1)))
    D2 = :(D2 = SVec{$N, $T}(get_data(S2)))
    r = :(r = D1 - D2)
    expr = Expr(:tuple, [:(r[$i]) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $D1
            $D2
            $r
            return $TensorType($expr)
        end
    end
end
@inline function sub{T <: SIMDTypes}(S1::Tensor{4, 3, T}, S2::Tensor{4, 3, T})
    @inbounds begin
        D1 = get_data(S1)
        D2 = get_data(S2)
        D180 = SVec{80, T}((D1[1],  D1[2],  D1[3],  D1[4],  D1[5],  D1[6],  D1[7],  D1[8],  D1[9],
                            D1[10], D1[11], D1[12], D1[13], D1[14], D1[15], D1[16], D1[17], D1[18],
                            D1[19], D1[20], D1[21], D1[22], D1[23], D1[24], D1[25], D1[26], D1[27],
                            D1[28], D1[29], D1[30], D1[31], D1[32], D1[33], D1[34], D1[35], D1[36],
                            D1[37], D1[38], D1[39], D1[40], D1[41], D1[42], D1[43], D1[44], D1[45],
                            D1[46], D1[47], D1[48], D1[49], D1[50], D1[51], D1[52], D1[53], D1[54],
                            D1[55], D1[56], D1[57], D1[58], D1[59], D1[60], D1[61], D1[62], D1[63],
                            D1[64], D1[65], D1[66], D1[67], D1[68], D1[69], D1[70], D1[71], D1[72],
                            D1[73], D1[74], D1[75], D1[76], D1[77], D1[78], D1[79], D1[80]))
        D280 = SVec{80, T}((D2[1],  D2[2],  D2[3],  D2[4],  D2[5],  D2[6],  D2[7],  D2[8],  D2[9],
                            D2[10], D2[11], D2[12], D2[13], D2[14], D2[15], D2[16], D2[17], D2[18],
                            D2[19], D2[20], D2[21], D2[22], D2[23], D2[24], D2[25], D2[26], D2[27],
                            D2[28], D2[29], D2[30], D2[31], D2[32], D2[33], D2[34], D2[35], D2[36],
                            D2[37], D2[38], D2[39], D2[40], D2[41], D2[42], D2[43], D2[44], D2[45],
                            D2[46], D2[47], D2[48], D2[49], D2[50], D2[51], D2[52], D2[53], D2[54],
                            D2[55], D2[56], D2[57], D2[58], D2[59], D2[60], D2[61], D2[62], D2[63],
                            D2[64], D2[65], D2[66], D2[67], D2[68], D2[69], D2[70], D2[71], D2[72],
                            D2[73], D2[74], D2[75], D2[76], D2[77], D2[78], D2[79], D2[80]))
        r = D180 - D280
        r81 = D1[81] - D2[81]
        return Tensor{4, 3}((r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                             r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                             r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                             r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                             r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                             r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                             r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                             r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                             r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r81))
    end
end

# *, /
@generated function mul{T <: SIMDTypes, TT <: AllSIMDTensors}(n::T, S::TT)
    TensorType = get_base(S)
    N = n_components(TensorType)
    D = :(D = SVec{$N, T}(get_data(S)))
    r = :(r = n * D)
    expr = Expr(:tuple, [:(r[$i]) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $D
            $r
            return $TensorType($expr)
        end
    end
end
@generated function mul{TT <: AllSIMDTensors, T <: SIMDTypes, }(S::TT, n::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    D = :(D = SVec{$N, T}(get_data(S)))
    r = :(r = D * n)
    expr = Expr(:tuple, [:(r[$i]) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $D
            $r
            return $TensorType($expr)
        end
    end
end
@generated function div{TT <: AllSIMDTensors, T <: SIMDTypes, }(S::TT, n::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    D = :(D = SVec{$N, T}(get_data(S)))
    r = :(r = D / n)
    expr = Expr(:tuple, [:(r[$i]) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds begin
            $D
            $r
            return $TensorType($expr)
        end
    end
end

@inline function mul{T <: SIMDTypes}(n::T, S::Tensor{4, 3, T})
    @inbounds begin
        D = get_data(S)
        D80 = SVec{80, T}((D[1],  D[2],  D[3],  D[4],  D[5],  D[6],  D[7],  D[8],  D[9],
                           D[10], D[11], D[12], D[13], D[14], D[15], D[16], D[17], D[18],
                           D[19], D[20], D[21], D[22], D[23], D[24], D[25], D[26], D[27],
                           D[28], D[29], D[30], D[31], D[32], D[33], D[34], D[35], D[36],
                           D[37], D[38], D[39], D[40], D[41], D[42], D[43], D[44], D[45],
                           D[46], D[47], D[48], D[49], D[50], D[51], D[52], D[53], D[54],
                           D[55], D[56], D[57], D[58], D[59], D[60], D[61], D[62], D[63],
                           D[64], D[65], D[66], D[67], D[68], D[69], D[70], D[71], D[72],
                           D[73], D[74], D[75], D[76], D[77], D[78], D[79], D[80]))
        r   = n * D80
        r81 = n * D[81]
        return Tensor{4, 3}((r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                             r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                             r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                             r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                             r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                             r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                             r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                             r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                             r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r81))
    end
end
@inline function mul{T <: SIMDTypes}(S::Tensor{4, 3, T}, n::T)
    @inbounds begin
        D = get_data(S)
        D80 = SVec{80, T}((D[1],  D[2],  D[3],  D[4],  D[5],  D[6],  D[7],  D[8],  D[9],
                           D[10], D[11], D[12], D[13], D[14], D[15], D[16], D[17], D[18],
                           D[19], D[20], D[21], D[22], D[23], D[24], D[25], D[26], D[27],
                           D[28], D[29], D[30], D[31], D[32], D[33], D[34], D[35], D[36],
                           D[37], D[38], D[39], D[40], D[41], D[42], D[43], D[44], D[45],
                           D[46], D[47], D[48], D[49], D[50], D[51], D[52], D[53], D[54],
                           D[55], D[56], D[57], D[58], D[59], D[60], D[61], D[62], D[63],
                           D[64], D[65], D[66], D[67], D[68], D[69], D[70], D[71], D[72],
                           D[73], D[74], D[75], D[76], D[77], D[78], D[79], D[80]))
        r   = D80 * n
        r81 = D[81] * n
        return Tensor{4, 3}((r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                             r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                             r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                             r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                             r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                             r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                             r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                             r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                             r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r81))
    end
end
@inline function div{T <: SIMDTypes}(S::Tensor{4, 3, T}, n::T)
    @inbounds begin
        D = get_data(S)
        D80 = SVec{80, T}((D[1],  D[2],  D[3],  D[4],  D[5],  D[6],  D[7],  D[8],  D[9],
                           D[10], D[11], D[12], D[13], D[14], D[15], D[16], D[17], D[18],
                           D[19], D[20], D[21], D[22], D[23], D[24], D[25], D[26], D[27],
                           D[28], D[29], D[30], D[31], D[32], D[33], D[34], D[35], D[36],
                           D[37], D[38], D[39], D[40], D[41], D[42], D[43], D[44], D[45],
                           D[46], D[47], D[48], D[49], D[50], D[51], D[52], D[53], D[54],
                           D[55], D[56], D[57], D[58], D[59], D[60], D[61], D[62], D[63],
                           D[64], D[65], D[66], D[67], D[68], D[69], D[70], D[71], D[72],
                           D[73], D[74], D[75], D[76], D[77], D[78], D[79], D[80]))
        r   = D80 / n
        r81 = D[81] / n
        return Tensor{4, 3}((r[1],  r[2],  r[3],  r[4],  r[5],  r[6],  r[7],  r[8],  r[9],
                             r[10], r[11], r[12], r[13], r[14], r[15], r[16], r[17], r[18],
                             r[19], r[20], r[21], r[22], r[23], r[24], r[25], r[26], r[27],
                             r[28], r[29], r[30], r[31], r[32], r[33], r[34], r[35], r[36],
                             r[37], r[38], r[39], r[40], r[41], r[42], r[43], r[44], r[45],
                             r[46], r[47], r[48], r[49], r[50], r[51], r[52], r[53], r[54],
                             r[55], r[56], r[57], r[58], r[59], r[60], r[61], r[62], r[63],
                             r[64], r[65], r[66], r[67], r[68], r[69], r[70], r[71], r[72],
                             r[73], r[74], r[75], r[76], r[77], r[78], r[79], r[80], r81))
    end
end


end # module
