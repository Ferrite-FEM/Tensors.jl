#################################################
# Sepcialized Second Order Symmetric Operations #
#################################################

Base.transpose(S::SymmetricTensors) = copy(S)


Base.issym(S::SymmetricTensors) = true

symmetrize(S::SymmetricTensors) = copy(S)
symmetrize!(t1::SymmetricTensors, t2::SymmetricTensors) = copy!(t1, t2)

######################
# Double contraction #
######################
@gen_code function dcontract{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    @code :(s = zero($Tv);
            data1 = get_data(S1);
            data2 = get_data(S2))
     for k in 1:n_independent_components(dim, true)
        if is_diagonal_index(dim, k)
            @code :(@inbounds s += data1[$k] * data2[$k])
        else
            @code :(@inbounds s += 2 * data1[$k] * data2[$k])
        end
    end
    @code :(return s)
end

function dcontract{dim, T1, T2}(S1::SymmetricTensor{4, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    t_new = zero(SymmetricTensor{2, dim, Tv})
    dcontract!(t_new, S1, S2)
end


function dcontract!{dim}(S::SymmetricTensor{2, dim}, S1::SymmetricTensor{4, dim},
                         S2::SymmetricTensor{2, dim})
    @inbounds for k in 1:dim, l in 1:k, i in 1:dim, j in 1:i
            S[i,j] += S1[i,j,k,l] * S2[k,l]
    end
    return S
end

function dcontract{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{4, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    t_new = zero(SymmetricTensor{2, dim, Tv})
    dcontract!(t_new, S1, S2)
end


function dcontract!{dim}(S::SymmetricTensor{2, dim}, S1::SymmetricTensor{2, dim},
                         S2::SymmetricTensor{4, dim})
    @inbounds for k in 1:dim, l in 1:k, i in 1:dim, j in i:dim
        S[i,j] += S2[i,j,k,l] * S1[i,j]
    end
    return S
end


########
# norm #
########
@gen_code function Base.norm{dim, T}(S::SymmetricTensor{4, dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(S), i, j)
    @code :(data = get_data(S))
    @code :(s = zero(T))
    for k in 1:dim, l in 1:k, i in 1:dim, j in 1:i
        @code :(@inbounds v = data[$(idx(i,j)), $(idx(k,l))])
        if i == j && k == l
             @code :(s += v*v)
        elseif i == j || k == l
             @code :(s += 2*v*v)
        else
             @code :(s += 4*v*v)
        end
    end
    @code :(return sqrt(s))
end


################
# Open product #
################
function otimes{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    S = SymmetricTensor{4, dim, Tv, 2}(zeros(Tv, length(get_data(S1)), length(get_data(S2))))
    otimes!(S, S1, S2)
end

function otimes!{dim}(S::SymmetricTensor{4, dim}, S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    A_mul_Bt!(get_data(S), get_data(S1), get_data(S2))
    return S
end
