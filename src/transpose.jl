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

@inline function Base.transpose(S::Tensor{2})
    typeof(S)(tomatrix(S).')
end
Base.transpose(S::SymmetricTensor{2}) = S

"""
```julia
minortranspose(::FourthOrderTensor)
```
Computes the minor transpose of a fourth order tensor.
"""
@generated function minortranspose{dim}(t::Tensor{4, dim})
    exps = Expr[]
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(exps, :(get_data(t)[$(compute_index(Tensor{4, dim}, j, i, l, k))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        Tensor{4, dim}($exp)
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
@generated function majortranspose{dim}(t::FourthOrderTensor{dim})
    exps = Expr[]
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(exps, :(get_data(t)[$(compute_index(get_base(t), k, l, i, j))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        Tensor{4, dim}($exp)
    end
end

Base.ctranspose(S::AllTensors) = transpose(S)
