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
    Tensor{2, dim}(Amt_mul_Bv(S1.data, S2.data))
end

@inline Base.(:*){dim}(S1::Tensor{4, dim}, S2::Tensor{2, dim}) = dcontract(S1, S2)
@inline Base.(:*){dim}(S1::Tensor{2, dim}, S2::Tensor{4, dim}) = dcontract(S1, S2)


########
# Norm #
########
@inline Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))
@inline Base.norm(S::Tensor{4}) = sqrt(sumabs2(get_data(S)))


################
# Open product #
################
@inline otimes{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = otimes(promote(S1, S2)...)

@inline function otimes{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
   Tensor{4, dim}(A_otimes_B(S1.data, S2.data))
end

@inline function otimes{dim}(v1::Vec{dim}, v2::Vec{dim})
    Tensor{2, dim}(A_otimes_B(v1.data, v2.data))
end

const ⊗ = otimes


symmetrize(t::AllTensors) = symmetrize!(similar(t), t)

function symmetrize!{dim}(t1::Tensor{2, dim}, t2::Tensor{2, dim})
    @assert get_base(typeof(t1)) == get_base(typeof(t2))
    for i in 1:dim, j in 1:i
        @inbounds v = 0.5 * (t2[i,j] + t2[j,i])
        t1[i,j] = v
        t1[j,i] = v
    end
    return t1
end


################
# Dot products #
################
@inline Base.dot{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = dot(promote(S1, S2)...)

@inline Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = A_dot_B(v1.data, v2.data)

@inline function Base.dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Am_mul_Bv(S1.data, v2.data))
end

@inline function Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::Tensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Amt_mul_Bv(S1.data, v2.data))
end


@inline function Base.dot{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return Tensor{2, dim}(Am_mul_Bm(S1.data, S2.data))
end

@inline function tdot{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return Tensor{2, dim}(Amt_mul_Bm(S1.data, S2.data))
end

@inline Base.(:*){dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = dot(S1, S2)

@inline function tdot{dim}(S2::Tensor{2, dim})
    return SymmetricTensor{2, dim}(transpdot(S2.data))
end

@inline Base.Ac_mul_B{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = dtot(promote(S1, S2)...)
@inline Base.Ac_mul_B{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = tdot(S1, S2)

@inline Base.At_mul_B{dim}(S1::AllTensors{dim}, S2::AllTensors{dim}) = tdot(promote(S1, S2)...)
@inline Base.At_mul_B{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = tdot(S1, S2)



#########
# Trace #
#########
@gen_code function LinAlg.trace{dim, T}(S::AllTensors{dim, T})
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
@gen_code function Base.det{dim, T}(t::SecondOrderTensor{dim, T})
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

@inline function Base.transpose{dim}(S::Tensor{2, dim})
    typeof(S)(mat_transpose(S.data))
end

Base.ctranspose(S::AllTensors) = transpose(S)
