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
2×2 Tensor{2, 2, Float64, 4}:
 0.325977  0.218587
 0.549051  0.894245

julia> A'
2×2 Tensor{2, 2, Float64, 4}:
 0.325977  0.549051
 0.218587  0.894245
```
"""
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

@inline Base.adjoint(S::AbstractTensor) = transpose(S)

@inline function Base.transpose(S::MixedTensor{2, dims}) where {dims}
    MixedTensor{2, (dims[2],dims[1])}(@inline function(i, j) @inbounds S[j,i]; end)
end

@inline function minortranspose(S::MixedTensor{4, dims}) where {dims}
    MixedTensor{4, (dims[2], dims[1], dims[4], dims[3])}(@inline function(i, j, k, l) @inbounds S[j,i,l,k]; end)
end

@inline function majortranspose(S::MixedTensor{4, dims}) where {dims}
    MixedTensor{4, (dims[3], dims[4], dims[1], dims[2])}(@inline function(i, j, k, l) @inbounds S[k,l,i,j]; end)
end
