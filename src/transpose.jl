# transpose, majortranspose, minortranspose
"""
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
@tensorop function Base.transpose(S::Union{Tensor{2}, MixedTensor{2}})
    C[i, j] = S[j, i]
end

@inline Base.transpose(S::SymmetricTensor{2}) = S

"""
    minortranspose(::FourthOrderTensor)

Compute the minor transpose of a fourth order tensor.
"""
@tensorop function minortranspose(S::Union{Tensor{4}, MixedTensor{4}})
    C[i, j, k, l] = S[j, i, l, k]
end

@inline minortranspose(S::SymmetricTensor{4}) = S
@inline Base.transpose(S::FourthOrderTensor) = minortranspose(S)

"""
    majortranspose(::FourthOrderTensor)

Compute the major transpose of a fourth order tensor.
The major transpose of a minor-symmetric tensor is again minor symmetric, so
a `SymmetricTensor` input gives a `SymmetricTensor` back.
"""
@tensorop function majortranspose(S::FourthOrderTensor)
    C[i, j, k, l] = S[k, l, i, j]
end

@inline Base.adjoint(S::AbstractTensor) = transpose(S)
