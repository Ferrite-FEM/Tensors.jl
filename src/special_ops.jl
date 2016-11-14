# specialized methods
"""
Computes a special dot product between two vectors and a symmetric fourth order tensor
such that ``a_k C_{ikjl} b_l``.

```julia
dotdot(::Vec, ::SymmetricFourthOrderTensor, ::Vec)
```
"""
@generated function dotdot{dim, T1, T2, T3}(v1::Vec{dim, T1}, S::SymmetricTensor{4, dim, T2}, v2::Vec{dim, T3})
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    exps = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
        for l in 1:dim, k in 1:dim
            push!(exps_ele.args, :(get_data(v1)[$k] * get_data(S)[$(idx(i,k,j,l))] * get_data(v2)[$l]))
        end
        push!(exps.args, exps_ele)
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds r = $exps
        Tensor{2, dim}(r)
    end
end
