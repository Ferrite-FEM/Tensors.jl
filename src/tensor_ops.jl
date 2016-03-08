######################
# Double contraction #
######################
@inline dcontract{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = dcontract(promote(S1, S2)...)

@inline function dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return A_dot_B(S1.data, S2.data)
end

@inline function dcontract{dim}(S1::Tensor{4, dim}, S2::Tensor{4, dim})
    Tensor{4, dim}(Am_mul_Bm(S1.data, S2.data))
end

@inline function dcontract{dim}(S1::Tensor{4, dim}, S2::Tensor{2, dim})
    Tensor{2, dim}(Am_mul_Bv(S1.data, S2.data))
end

@inline function dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{4, dim})
    Tensor{2, dim}(Amt_mul_Bv(S2.data, S1.data))
end

@inline Base.(:*){dim}(S1::Tensor{4, dim}, S2::Tensor{2, dim}) = dcontract(S1, S2)
@inline Base.(:*){dim}(S1::Tensor{2, dim}, S2::Tensor{4, dim}) = dcontract(S1, S2)
@inline Base.(:*){dim}(S1::SymmetricTensor{4, dim}, S2::SymmetricTensor{2, dim}) = dcontract(S1, S2)
@inline Base.(:*){dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{4, dim}) = dcontract(S1, S2)

const ⊡ = dcontract

########
# Norm #
########
@inline Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))
@inline Base.norm(S::Tensor{4}) = sqrt(sumabs2(get_data(S)))


################
# Open product #
################
"""
Computes the outer product between two tensors `t1` and `t2`. Can also be called via the infix operator `⊗`.
"""
@inline otimes{order, dim}(t1::AbstractTensor{order, dim}, t2::AbstractTensor{order, dim}) = otimes(promote(t1, t2)...)

@inline function otimes{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
   Tensor{4, dim}(A_otimes_B(S1.data, S2.data))
end

@inline function otimes{dim}(v1::Vec{dim}, v2::Vec{dim})
    Tensor{2, dim}(A_otimes_B(v1.data, v2.data))
end

const ⊗ = otimes


################
# Dot products #
################
import Base.dot



@inline dot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = A_dot_B(v1.data, v2.data)

@inline function dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Am_mul_Bv(S1.data, v2.data))
end

@inline function dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::Tensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Amt_mul_Bv(S2.data, v1.data))
end

@inline Base.(:*){dim}(S1::Tensor{1, dim}, S2::Tensor{2, dim}) = dot(S1, S2)
@inline Base.(:*){dim}(S1::Tensor{2, dim}, S2::Tensor{1, dim}) = dot(S1, S2)

@inline function dot{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return Tensor{2, dim}(Am_mul_Bm(S1.data, S2.data))
end

@inline function tdot{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return Tensor{2, dim}(Amt_mul_Bm(S1.data, S2.data))
end

@inline function dot{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    S1_t = convert(Tensor{2, dim}, S1)
    S2_t = convert(Tensor{2, dim}, S2)
    return Tensor{2, dim}(Am_mul_Bm(S1_t.data, S2_t.data))
end

@inline Base.(:*){dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = dot(S1, S2)

@inline function tdot{dim}(S2::Tensor{2, dim})
    return SymmetricTensor{2, dim}(transpdot(S2.data))
end

@inline Base.Ac_mul_B{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = tdot(promote(S1, S2)...)
@inline Base.Ac_mul_B{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = tdot(S1, S2)

@inline Base.At_mul_B{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = tdot(promote(S1, S2)...)
@inline Base.At_mul_B{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = tdot(S1, S2)



#########
# Trace #
#########
import Base.LinAlg.trace
"""
Computes the trace of a second or fourth order tensor.
"""
@gen_code function trace{dim, T}(S::Union{SecondOrderTensor{dim, T}, FourthOrderTensor{dim, T}})
    @code :($(Expr(:meta, :inline)))
    @code :(s = zero(T))
    for i = 1:dim
        if S <: SecondOrderTensor
            @code :(@inbounds s += S[$i,$i])
        elseif S <: FourthOrderTensor
            @code :(@inbounds s += S[$i,$i,$i,$i])
        end
    end
    @code :(return s)
end


############
# Deviator #
############
#@gen_code function dev{dim}(S1::SecondOrderTensor{dim})
#    idx(i,j) = compute_index(S, i, j)
#    @code :(vol = mean(S1))
#    @code :(data = get_data(S))
#     for i = 1:dim, j = 1:dim
#        if i == j
#            @code :(data[$(idx(i,j))] -= vol)
#        end
#    end
#    @code :(return  S)
#end


########
# Mean #
########
Base.mean{dim}(S::SecondOrderTensor{dim}) = trace(S) / dim



###############
# Determinant #
###############
import Base.det

"""
Computes the trace of a second order tensor.
"""
@gen_code function det{dim, T}(t::SecondOrderTensor{dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :(v = get_data(t))
    if dim == 1
        @code :(@inbounds d = v[$(idx(1,1))])
    elseif dim == 2
        @code :(@inbounds d = v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))] * v[$(idx(2,1))])
    else
        @code :(@inbounds d = ( v[$(idx(1,1))]*(v[$(idx(2,2))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,2))]) -
                                v[$(idx(1,2))]*(v[$(idx(2,1))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,1))]) +
                                v[$(idx(1,3))]*(v[$(idx(2,1))]*v[$(idx(3,2))]-v[$(idx(2,2))]*v[$(idx(3,1))])))
    end
    @code :(return d)
end


###########
# Inverse #
###########

import Base.inv

"""
Computes the inverse of a second order tensor.
"""
@gen_code function Base.inv{dim, T}(t::Tensor{2, dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return  typeof(t)((dinv,)))
    elseif dim == 2
        @code :( return typeof(t)((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                  -v[$(idx(1,2))] * dinv, v[$(idx(1,1))] * dinv)))
    else
        @code :(return typeof(t)((  (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                                   -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                                    (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                                   -(v[$(idx(1,2))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,2))]) * dinv,
                                    (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,2))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,2))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,1))]) * dinv,
                                    (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv)))
    end
end

"""
Computes the inverse of a second order symmetric tensor.
"""
@gen_code function Base.inv{dim, T}(t::SymmetricTensor{2, dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return  typeof(t)((dinv,)))
    elseif dim == 2
        @code :( return typeof(t)((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                   v[$(idx(1,1))] * dinv)))
    else
        @code :(return typeof(t)((  (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                                   -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                                    (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv)))
    end
end


#############
# Transpose #
#############

"""
Computes the transpose of a second order tensor.
"""
@inline function Base.transpose{dim}(S::Tensor{2, dim})
    typeof(S)(mat_transpose(S.data))
end

Base.ctranspose(S::AllTensors) = transpose(S)
