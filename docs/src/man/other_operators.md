```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
end
```

# Other operators

```@index
Pages = ["other_operators.md"]
```

## Transpose-dot

The dot product between the transpose of a tensor with itself. Results in a symmetric tensor.

```math
\mathbf{A} = \mathbf{B}^\text{T} \cdot \mathbf{B} \Leftrightarrow A_{ij} = B_{ki}^\text{T} B_{kj} = B_{ik} B_{kj}
```

```math
\mathbf{A} = \mathbf{B} \cdot \mathbf{B}^\text{T} \Leftrightarrow A_{ij} = B_{ik} B_{jk}^\text{T} = B_{ik} B_{kj}
```

```@docs
Tensors.tdot
Tensors.dott
```

## Norm

The (2)-norm of a tensor is defined for a vector, second order tensor and fourth order tensor as

```math
\|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}} \Leftrightarrow \|a_i\| = \sqrt{a_i a_i},
```

```math
\|\mathbf{A}\| = \sqrt{\mathbf{A} : \mathbf{A}} \Leftrightarrow \|A_{ij}\| = \sqrt{A_{ij} A_{ij}},
```

```math
\|\mathsf{A}\| = \sqrt{\mathsf{A} :: \mathsf{A}} \Leftrightarrow \|A_{ijkl}\| = \sqrt{A_{ijkl} A_{ijkl}}.
```

```@docs
Tensors.norm
```

## Trace

The trace for a second order tensor is defined as the sum of the diagonal elements. This can be written as

```math
\text{tr}(\mathbf{A}) = \mathbf{I} : \mathbf{A} \Leftrightarrow \text{tr}(A_{ij}) = A_{ii}.
```

```@docs
Tensors.tr
```

## Determinant

Determinant for a second order tensor.

```@docs
Tensors.det
```

## Inverse

Inverse of a second order tensor such that

```math
\mathbf{A}^{-1} \cdot \mathbf{A} = \mathbf{I}
```

where ``\mathbf{I}`` is the second order identity tensor.

```@docs
Tensors.inv
```

## Transpose

Transpose of tensors is defined by changing the order of the tensor's "legs". The transpose of a vector/symmetric tensor is the vector/tensor itself. The transpose of a second order tensor can be written as:

```math
A_{ij}^\text{T} = A_{ji}
```

and for a fourth order tensor the minor transpose can be written as

```math
A_{ijkl}^\text{t} = A_{jilk}
```

and the major transpose as

```math
A_{ijkl}^\text{T} = A_{klij}.
```

```@docs
Tensors.transpose
Tensors.minortranspose
Tensors.majortranspose
```

## Symmetric

The symmetric part of a second order tensor is defined by:

```math
\mathbf{A}^\text{sym} = \frac{1}{2}(\mathbf{A} + \mathbf{A}^\text{T}) \Leftrightarrow A_{ij}^\text{sym} = \frac{1}{2}(A_{ij} + A_{ji}),
```

The major symmetric part of a fourth order tensor is defined by
```math
\mathsf{A}^\text{majsym} = \frac{1}{2}(\mathsf{A} + \mathsf{A}^\text{T}) \Leftrightarrow A_{ijkl}^\text{majsym} = \frac{1}{2}(A_{ijkl} + A_{klij}).
```

The minor symmetric part of a fourth order tensor is defined by 
```math
A_{ijkl}^\text{minsym} = \frac{1}{4}(A_{ijkl} + A_{ijlk} + A_{jikl} + A_{jilk}).
```

```@docs
Tensors.symmetric
Tensors.minorsymmetric
Tensors.majorsymmetric
```

## Skew symmetric

The skew symmetric part of a second order tensor is defined by

```math
\mathbf{A}^\text{skw} = \frac{1}{2}(\mathbf{A} - \mathbf{A}^\text{T}) \Leftrightarrow A^\text{skw}_{ij} = \frac{1}{2}(A_{ij} - A_{ji}).
```

The skew symmetric part of a symmetric tensor is zero.

```@docs
Tensors.skew
```

## Deviatoric tensor

The deviatoric part of a second order tensor is defined by

```math
\mathbf{A}^\text{dev} = \mathbf{A} - \frac{1}{3} \mathrm{tr}[\mathbf{A}] \mathbf{I} \Leftrightarrow A_{ij}^\text{dev} = A_{ij} - \frac{1}{3}A_{kk}\delta_{ij}.
```

```@docs
Tensors.dev
```

## Volumetric tensor

The volumetric part of a second order tensor is defined by

```math
\mathbf{A}^\text{vol} = \frac{1}{3} \mathrm{tr}[\mathbf{A}] \mathbf{I} \Leftrightarrow A_{ij}^\text{vol} = \frac{1}{3}A_{kk}\delta_{ij}.
```

```@docs
Tensors.vol
```

## Cross product

The cross product between two vectors is defined as

```math
\mathbf{a} = \mathbf{b} \times \mathbf{c} \Leftrightarrow a_i = \epsilon_{ijk} b_j c_k
```

```@docs
Tensors.cross
```

## Eigenvalues and eigenvectors

The eigenvalues and eigenvectors of a (symmetric) second order tensor, ``\mathbf{A}`` can be solved from the eigenvalue problem

```math
\mathbf{A} \cdot \mathbf{v}_i = \lambda_i \mathbf{v}_i \qquad i = 1, \dots, \text{dim}
```

where ``\lambda_i`` are the eigenvalues and ``\mathbf{v}_i`` are the corresponding eigenvectors.
For a symmetric fourth order tensor, ``\mathsf{A}`` the second order eigentensors and eigenvalues
can be solved from

```math
\mathsf{A} : \mathbf{V}_i = \lambda_i \mathbf{V}_i \qquad i = 1, \dots, \text{dim}
```

where ``\lambda_i`` are the eigenvalues and ``\mathbf{V}_i`` the corresponding eigentensors.

```@docs
Tensors.eigen
Tensors.eigvals
Tensors.eigvecs
```

## Tensor square root

Square root of a symmetric positive definite second order tensor ``S``,
defined such that

```math
\sqrt{\mathbf{S}} \cdot \sqrt{\mathbf{S}} = S.
```

```@docs
Tensors.sqrt
```

## Rotations

```@docs
Tensors.rotate
Tensors.rotation_tensor
```

## Special operations

For computing a special dot product between two vectors ``\mathbf{a}`` and ``\mathbf{b}`` with a fourth order symmetric tensor ``\mathbf{C}`` such that ``a_k C_{ikjl} b_l`` there is `dotdot(a, C, b)`. This function is useful because it is the expression for the tangent matrix in continuum mechanics when the displacements are approximated by scalar shape functions.

```@docs
Tensors.dotdot
```

## Voigt format

For some operations it is convenient to easily switch to the so called "Voigt"-format.
For example when solving a local problem in a plasticity model. To simplify the conversion
between tensors and Voigt format, see [`tovoigt`](@ref), [`tovoigt!`](@ref) and
[`fromvoigt`](@ref) documented below. Care must be exercised when combined with
differentiation, see [Differentiation of Voigt format](@ref) further down.

```@docs
Tensors.tovoigt
Tensors.tovoigt!
Tensors.fromvoigt
```

### Differentiation of Voigt format

Differentiating with a Voigt representation of a symmetric tensor may lead to incorrect
results when converted back to tensors.
The `tomandel`, `tomandel!`, and `frommandel` versions of
[`tovoigt`](@ref), [`tovoigt!`](@ref), and [`fromvoigt`](@ref) can then be used. As
illustrated by the following example, this will give the correct result. In general,
however, direct differentiation of `Tensor`s is faster (see
[Automatic Differentiation](@ref)).

```jldoctest ad-voigt
julia> using Tensors, ForwardDiff

julia> fun(X::SymmetricTensor{2}) = X;

julia> A = rand(SymmetricTensor{2,2});
```

Differentiation of a tensor directly (correct):
```jldoctest ad-voigt
julia> tovoigt(gradient(fun, A))
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.5
```

Converting to Voigt format, perform differentiation, convert back (WRONG!):
```jldoctest ad-voigt
julia> ForwardDiff.jacobian(
           v -> tovoigt(fun(fromvoigt(SymmetricTensor{2,2}, v))),
           tovoigt(A)
       )
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```

Converting to Mandel format, perform differentiation, convert back (correct)
```jldoctest ad-voigt
julia> tovoigt(
           frommandel(SymmetricTensor{4,2},
                ForwardDiff.jacobian(
                    v -> tomandel(fun(frommandel(SymmetricTensor{2,2}, v))),
                    tomandel(A)
                )
           )
       )
3×3 Matrix{Float64}:
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  0.5
```
