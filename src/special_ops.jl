# specialized methods
"""
```julia
dotdot(::Vec, ::SymmetricFourthOrderTensor, ::Vec)
```
Computes a special dot product between two vectors and a symmetric fourth order tensor
such that ``a_k C_{ikjl} b_l``.
"""
@generated function dotdot{dim}(v1::Vec{dim}, S::SymmetricTensor{4, dim}, v2::Vec{dim})
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    exps = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        ex1, ex2 = Expr[],Expr[]
        for l in 1:dim, k in 1:dim
            push!(ex1, :(get_data(v1)[$k] * get_data(S)[$(idx(i,k,j,l))]))
            push!(ex2, :(get_data(v2)[$l]))
        end
        push!(exps.args, reducer(ex1, ex2, true))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{2, dim}($exps)
    end
end


function tovoigt{dim,T,M}(A::SymmetricTensor{2,dim,T,M}; offdiagscale::Real=1)
    v = zeros(T, M)
    for i in 1:dim, j in i:dim
        idx, scale = _index_scale(dim, i, j, offdiagscale)
        v[idx] = scale * A[i,j]
    end
    return v
end

function tovoigt{dim,T,M}(A::SymmetricTensor{4,dim,T,M}; offdiagscale::Real=1)
    n = Int(√M)
    v = zeros(T, n, n)
    for i in 1:dim, j in i:dim, k in 1:dim, l in k:dim
        I, si = _index_scale(dim, i, j, offdiagscale)
        J, sj = _index_scale(dim, k, l, offdiagscale)
        v[I,J] = si * sj * A[i,j,k,l]
    end
    return v
end

function tovoigt{dim,T,M}(A::Tensor{2,dim,T,M}; offdiagscale::Real=1)
    v = zeros(T, M)
    for i in 1:dim, j in 1:dim
        idx, scale = _index_scale(dim, i, j, offdiagscale)
        v[idx] = scale * A[i,j]
    end
    return v
end

function tovoigt{dim,T,M}(A::Tensor{4,dim,T,M}; offdiagscale::Real=1)
    n = Int(√M)
    v = zeros(T, n, n)
    for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim
        I, si = _index_scale(dim, i, j, offdiagscale)
        J, sj = _index_scale(dim, k, l, offdiagscale)
        v[I,J] = si * sj * A[i,j,k,l]
    end
    return v
end

# Get index and scale to reduce order of symmetric tensor
#
# Example
# index:
# [1 6 5
#  9 2 4
#  8 7 3]
#
# scale:
# [1 s s
#  s 1 s
#  s s 1]
#
function _index_scale(dim::Int, i::Int, j::Int, s::Real) # i ≤ dim and j ≤ dim are assumed
    if i == j
        (i, one(typeof(s)))
    elseif i < j
        (_offdiagind(dim, i, j), s)
    else
        (_offdiagind(dim, j, i) + sum(1:dim-1), s)
    end
end

function _offdiagind(dim::Int, i::Int, j::Int) # i < j ≤ dim is assumed
    count = dim + (j-1) - (i-1)
    for idx in j+1:dim
        count += idx-1
    end
    count
end
