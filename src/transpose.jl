# transpose, majortranspose, minortranspose
"""
```julia
transpose(::Vec)
transpose(::SecondOrderTensor)
transpose(::FourthOrderTensor)
```
Computes the transpose of a tensor.
For a fourth order tensor, the transpose is the minor transpose.

**Example:**

```jldoctest
julia> A = rand(Tensor{2,2})
2×2 Tensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> A'
2×2 Tensors.Tensor{2,2,Float64,4}:
 0.590845  0.766797
 0.566237  0.460085
```
"""
@inline Base.transpose(S::Vec) = S

@generated function Base.transpose{dim}(S::Tensor{2, dim})
    expr = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        push!(expr.args, :(get_data(S)[$(compute_index(Tensor{2, dim}, j, i))]))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{2, dim}($expr)
    end
end

Base.transpose(S::SymmetricTensor{2}) = S

"""
```julia
minortranspose(::FourthOrderTensor)
```
Computes the minor transpose of a fourth order tensor.
"""
@generated function minortranspose{dim}(S::Tensor{4, dim})
    expr = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(expr.args, :(get_data(S)[$(compute_index(Tensor{4, dim}, j, i, l, k))]))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($expr)
    end
end

minortranspose(S::SymmetricTensor{4}) = S
Base.transpose(S::FourthOrderTensor) = minortranspose(S)

"""
```julia
majortranspose(::FourthOrderTensor)
```
Computes the major transpose of a fourth order tensor.
"""
@generated function majortranspose{dim}(S::FourthOrderTensor{dim})
    expr = Expr(:tuple)
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(expr.args, :(get_data(S)[$(compute_index(get_base(S), k, l, i, j))]))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds return Tensor{4, dim}($expr)
    end
end

Base.ctranspose(S::AllTensors) = transpose(S)
