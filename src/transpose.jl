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

@inline function Base.transpose{dim}(S::Tensor{2, dim})
    Tensor{2, dim}(@inline function(i, j) @inboundsret S[j,i]; end)
end

@inline Base.transpose(S::SymmetricTensor{2}) = S

"""
```julia
minortranspose(::FourthOrderTensor)
```
Computes the minor transpose of a fourth order tensor.
"""
@inline function minortranspose{dim}(S::Tensor{4, dim})
    Tensor{4, dim}(@inline function(i, j, k, l) @inboundsret S[j,i,l,k]; end)
end

@inline minortranspose(S::SymmetricTensor{4}) = S
@inline Base.transpose(S::FourthOrderTensor) = minortranspose(S)

"""
```julia
majortranspose(::FourthOrderTensor)
```
Computes the major transpose of a fourth order tensor.
"""
@inline function majortranspose{dim}(S::FourthOrderTensor{dim})
    Tensor{4, dim}(@inline function(i, j, k, l) @inboundsret S[k,l,i,j]; end)
end

@inline Base.ctranspose(S::AllTensors) = transpose(S)
