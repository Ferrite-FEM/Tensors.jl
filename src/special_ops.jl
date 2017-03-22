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


function tovoigt{dim,T,M}(A::SymmetricTensor{2,dim,T,M}, c::Real)
    v = zeros(T, M)
    for i in 1:dim, j in i:dim
        idx, coef = _index_coef(dim, i, j, c)
        v[idx] = coef * A[i,j]
    end
    return v
end

function tovoigt{dim,T,M}(A::SymmetricTensor{4,dim,T,M}, c::Real)
    n = Int(√M)
    v = zeros(T, n, n)
    for i in 1:dim, j in i:dim, k in 1:dim, l in k:dim
        I, ci = _index_coef(dim, i, j, c)
        J, cj = _index_coef(dim, k, l, c)
        v[I,J] = ci * cj * A[i,j,k,l]
    end
    return v
end

# Get index and coefficient to reduce order of symmetric tensor
#
# Example
# index:
# [1 6 5
#  ⋅ 2 4
#  ⋅ ⋅ 3]
#
# coefficient:
# [1  c  c
#  ⋅  1  c
#  ⋅  ⋅  1]
#
function _index_coef(dim::Int, i::Int, j::Int, c::Real) # i ≤ j ≤ dim is assumed
    if i == j
        (i, one(typeof(c)))
    else
        (_offdiagind(dim, i, j), c)
    end
end

function _offdiagind(dim::Int, i::Int, j::Int) # i < j ≤ dim is assumed
    count = dim + (j-1) - (i-1)
    for idx in j+1:dim
        count += idx-1
    end
    count
end
