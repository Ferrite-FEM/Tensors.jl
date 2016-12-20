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
        exps_ele = Expr[]
        for l in 1:dim, k in 1:dim
            push!(exps_ele, :(get_data(v1)[$k] * get_data(S)[$(idx(i,k,j,l))] * get_data(v2)[$l]))
        end
        push!(exps.args, reduce((ex1, ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds r = $exps
        Tensor{2, dim}(r)
    end
end
