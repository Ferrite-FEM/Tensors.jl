# transpose, majortranspose, minortranspose
"""
    transpose(::Vec)
    transpose(::SecondOrderTensor)
    transpose(::FourthOrderTensor)

Compute the transpose of a tensor.
For a fourth order tensor, the transpose is the minor transpose.

# Examples
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

@inline function Base.transpose(S::Tensor{2, dim}) where {dim}
    Tensor{2, dim}(@inline function(i, j) @inbounds S[j,i]; end)
end

@inline Base.transpose(S::SymmetricTensor{2}) = S

"""
    minortranspose(::FourthOrderTensor)

Compute the minor transpose of a fourth order tensor.
"""
@inline function minortranspose(S::Tensor{4, dim}) where {dim}
    Tensor{4, dim}(@inline function(i, j, k, l) @inbounds S[j,i,l,k]; end)
end

@inline minortranspose(S::SymmetricTensor{4}) = S
@inline Base.transpose(S::FourthOrderTensor) = minortranspose(S)

"""
    majortranspose(::FourthOrderTensor)

Compute the major transpose of a fourth order tensor.
"""
@inline function majortranspose(S::FourthOrderTensor{dim}) where {dim}
    Tensor{4, dim}(@inline function(i, j, k, l) @inbounds S[k,l,i,j]; end)
end

if VERSION >= v"0.7.0-DEV.1415"
    @inline Base.adjoint(S::AllTensors) = transpose(S)
else
    @inline Base.ctranspose(S::AllTensors) = transpose(S)
end
