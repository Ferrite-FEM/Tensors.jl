# Multilinear forms: named evaluation of products of three tensors.
"""
    dotdot(::Vec, ::SecondOrderTensor, ::Vec)
    dotdot(::Vec, ::FourthOrderTensor, ::Vec)
    dotdot(::SecondOrderTensor, ::FourthOrderTensor, ::SecondOrderTensor)

Evaluate a bilinear form. The three methods compute, respectively,

```math
a_i S_{ij} b_j, \\quad a_k \\mathsf{C}_{ikjl} b_l, \\quad \\mathbf{A}_{ij} \\mathsf{C}_{ijkl} \\mathbf{B}_{kl}
```

The vector–fourth-order–vector form (a second-order tensor) is the tangent
stiffness contribution for scalar shape functions in continuum mechanics; the
second-order–fourth-order–second-order form is the corresponding
energy/stiffness integrand `A ⊡ C ⊡ B`.

# Examples
```jldoctest
julia> a = rand(Vec{2}); S = rand(SymmetricTensor{2,2});

julia> dotdot(a, S, a) ≈ a ⋅ (S ⋅ a)
true
```
"""
function dotdot end

@tensorop function dotdot(v1::AbstractTensor{1}, S::SecondOrderTensor, v2::AbstractTensor{1})
    @muladd C = v1[i] * S[i, j] * v2[j]
end

@tensorop function dotdot(v1::AbstractTensor{1}, S::FourthOrderTensor, v2::AbstractTensor{1})
    @muladd C[i, j] = v1[k] * S[i, k, j, l] * v2[l]
end

# Evaluated pairwise, not as a fused ternary kernel: the two dcontract SIMD
# kernels beat the scalar lowering of the fused form (measured 3.5 vs 5.4 ns
# for symmetric arguments at dim 3, 6.1 vs 15.5 ns for full ones).
@inline function dotdot(A::SecondOrderTensor, C::FourthOrderTensor, B::SecondOrderTensor)
    return dcontract(dcontract(A, C), B)
end
