######################
# Double contraction #
######################
function dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return dot(vec(get_data(S1)), vec(get_data(S2)))
end

function dcontract!{dim, T, T1, T2}(S::Tensor{4, dim, T}, S1::Tensor{4, dim, T1}, S2::Tensor{4, dim, T2})
    A_mul_B!(get_data(S), get_data(S1), get_data(S2))
    return S
end

function dcontract{dim}(S1::Tensor{4, dim}, S2::Tensor{4, dim})
    Tv = typeof(zero(T1) * zero(T2))
    dcontract!(zero(Tensor{4, dim, Tv}), S1, S2)
end

Base.(:*)(S1::AbstractTensor, S2::AbstractTensor) = dcontract(S1, S2)


########
# norm #
########
Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))
Base.norm(S::Tensor{4}) = sqrt(sumabs2(get_data(S)))


########
# Norm #
########

################
# Open product #
################
function otimes{dim, T1, T2}(S1::Tensor{2, dim, T1}, S2::Tensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    S = Tensor{4, dim, Tv, 2}(zeros(Tv, length(get_data(S1)), length(get_data(S2))))
    otimes!(S, S1, S2)
end

function otimes!{dim}(S::Tensor{4, dim}, S1::Tensor{2, dim}, S2::Tensor{2, dim})
    A_mul_Bt!(get_data(S), get_data(S1), get_data(S2))
    return S
end

function otimes{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    n = n_independent_components(Tensor{2, dim, Tv})
    S = Tensor{2, dim, Tv, 1}(zeros(Tv, n))
    otimes_unsym!(S, v1, v2)
end

@gen_code function otimes!{dim}(S::Tensor{2, dim}, v1::Vec{dim}, v2::Vec{dim})
    idx(i,j) = compute_index(S, i, j)
    @code :(data = get_data(S))
    for i = 1:dim, j = 1:dim
        @code :(@inbounds data[$(idx(i,j))] = v1[$i] * v2[$j])
    end
    @code :(return S)
end

const ⊗ = otimes


################
# Dot products #
################
function Base.dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = eltype(zero(T1) * zero(T2))
    v = zero(Vec{dim, Tv})
    dot!(v, S1, v2)
end

function dot!{dim}(v::Vec{dim}, S1::Tensor{2, dim}, v2::Vec{dim})
    data_matrix_1 = reshape(get_data(S1), size(S1))
    data_vec_2 = get_data(v2)
    data_vec =  get_data(v)
    A_mul_B!(data_vec, data_matrix_1, data_vec_2)
    return v
end

function Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::Tensor{2, dim, T2})
    Tv = eltype(zero(T1) * zero(T2))
    v = zero(Vec{dim, Tv})
    dot!(v, v1, S2)
end

function dot!{dim}(v::Vec{dim},  v1::Vec{dim}, S2::Tensor{2, dim},)
    data_matrix_1 = reshape(get_data(S2), size(S2))
    data_vec_2 = get_data(v1)
    data_vec =  get_data(v)
    At_mul_B!(data_vec, data_matrix_1, data_vec_2)
    return v
end

function Base.dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, S2::Tensor{2, dim, T2})
    Tv = eltype(zero(T1) * zero(T2))
    S = zero(Tensor{2, dim, Tv})
    dot!(S, S1, S2)
end

function dot!{dim}(S::Tensor{2, dim}, S1::Tensor{2, dim}, S2::Tensor{2, dim})
    data_matrix_1 = reshape(get_data(S1), size(S1))
    data_matrix_2 = reshape(get_data(S2), size(S2))
    data_matrix = reshape(get_data(S), size(S))
    A_mul_B!(data_matrix, data_matrix_1, data_matrix_2)
    return S
end

#########
# Trace #
#########
@gen_code function LinAlg.trace{dim, T}(S::AllTensors{dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(S), i, j)
    @code :(s = zero(T))
    @code :(v = get_data(S))
    for i = 1:dim
        if S <: SecondOrderTensor
            @code :(@inbounds s += v[$(idx(i,i))])
        elseif S <: FourthOrderTensor
            @code :(@inbounds s += v[$(idx(i,i)), $(idx(i,i))])
        end
    end
    @code :(return s)
end

############
# Deviator #
############
@gen_code function dev!{dim}(S::SecondOrderTensor{dim}, S1::SecondOrderTensor{dim})
    idx(i,j) = compute_index(S, i, j)
    @code :(copy!(S, S1))
    @code :(vol = mean(S1))
    @code :(data = get_data(S))
     for i = 1:dim, j = 1:dim
        if i == j
            @code :(data[$(idx(i,j))] -= vol)
        end
    end
    @code :(return  S)
end

dev(S::SecondOrderTensor) = dev!(similar(S), S)

########
# Mean #
########
Base.mean{dim}(S::SecondOrderTensor{dim}) = trace(S) / dim


###############
# Determinant #
###############
@gen_code function det{dim, T, M}(S::Tensor{2, dim, T, M})
    idx(i,j) = compute_sym_index(S, i, j)
    @code :(v = get_data(S))
    if dim == 1
        @code :(@inbounds d = v[1])
    elseif dim == 2
        @code :(@inbounds d = v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))] * v[$(idx(2,1))])
    else
        @code :(@inbounds d = ( v[$(idx(1,1))]*(v[$(idx(2,2))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,2))]) -
                                v[$(idx(1,2))]*(v[$(idx(2,1))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,1))]) +
                                v[$(idx(1,3))]*(v[$(idx(2,1))]*v[$(idx(3,2))]-v[$(idx(2,2))]*v[$(idx(3,1))])))
    end
    @code :(return d)
end

#############
# Transpose #
#############
function Base.transpose{dim}(S::Tensor{2, dim})
    S_new = copy(S)
    @inbounds for i in 1:dim, j in 1:i
        S_new[i,j] = S[j,i]
        S_new[j,i] = S[i,j]
    end
    return S_new
end

function Base.ctranspose{dim}(S::Tensor{2, dim})
    S_new = copy(S)
    @inbounds for i in 1:dim, j in 1:i
        S_new[i,j] = conj(S[j,i])
        S_new[j,i] = conj(S[i,j])
    end
    return S_new
end

