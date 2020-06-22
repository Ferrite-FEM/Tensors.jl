# specialized methods
"""
    dotdot(::Vec, ::SymmetricFourthOrderTensor, ::Vec)

Computes a special dot product between two vectors and a symmetric fourth order tensor
such that ``a_k C_{ikjl} b_l``.
"""
@generated function dotdot(v1::Vec{dim}, S::SymmetricTensor{4, dim}, v2::Vec{dim}) where {dim}
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    exps = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        ex1, ex2 = Expr[],Expr[]
        for l in 1:dim, k in 1:dim
            push!(ex1, :(Tuple(v1)[$k] * Tuple(S)[$(idx(i,k,j,l))]))
            push!(ex2, :(Tuple(v2)[$l]))
        end
        push!(exps.args, reducer(ex1, ex2, true))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{2, dim}($exps)
    end
end
