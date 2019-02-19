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

where ``\mathbf{I}`` is the second order identitiy tensor.

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

The symmetric part of a second and fourth order tensor is defined by:

```math
\mathbf{A}^\text{sym} = \frac{1}{2}(\mathbf{A} + \mathbf{A}^\text{T}) \Leftrightarrow A_{ij}^\text{sym} = \frac{1}{2}(A_{ij} + A_{ji}),
```
```math
\mathsf{A}^\text{sym} = \frac{1}{2}(\mathsf{A} + \mathsf{A}^\text{t}) \Leftrightarrow A_{ijkl}^\text{sym} = \frac{1}{2}(A_{ijkl} + A_{jilk}).
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
```

## Special operations

For computing a special dot product between two vectors ``\mathbf{a}`` and ``\mathbf{b}`` with a fourth order symmetric tensor ``\mathbf{C}`` such that ``a_k C_{ikjl} b_l`` there is `dotdot(a, C, b)`. This function is useful because it is the expression for the tangent matrix in continuum mechanics when the displacements are approximated by scalar shape functions.

```@docs
Tensors.dotdot
```

## Voigt

```@docs
Tensors.tovoigt
Tensors.fromvoigt
```
