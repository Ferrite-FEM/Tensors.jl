# transpose, majortranspose, minortranspose
"""
Computes the transpose of a tensor.
For a fourth order tensor, the transpose is the minor transpose

```julia
transpose(::Vec)
transpose(::SecondOrderTensor)
transpose(::FourthOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> A'
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
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
Computes the minor transpose of a fourth order tensor.

```julia
minortranspose(::FourthOrderTensor)
```
"""
@generated function minortranspose{dim, T, N}(t::Tensor{4, dim, T, N})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
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
Computes the major transpose of a fourth order tensor.

```julia
majortranspose(::FourthOrderTensor)
```
"""
@generated function majortranspose{dim, T}(t::FourthOrderTensor{dim, T})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        push!(exps, :(get_data(t)[$(compute_index(get_base(t), k, l, i, j))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        Tensor{4, dim}($exp)
    end
end

Base.ctranspose(S::AllTensors) = transpose(S)
