```@meta
DocTestSetup = quote
    srand(1234)
    using Tensors
end
```

# Other operators

```@index
Pages = ["other_operators.md"]
```

## Transpose-dot
The dot product between the transpose of a tensor with itself. Results in a symmetric tensor.

$\mathbf{A} = \mathbf{B}^\text{T} \cdot \mathbf{B} \Leftrightarrow A_{ij} = B_{ki}^\text{T} B_{kj} = B_{ik} B_{kj}$

$\mathbf{A} = \mathbf{B} \cdot \mathbf{B}^\text{T} \Leftrightarrow A_{ij} = B_{ik} B_{jk}^\text{T} = B_{ik} B_{kj}$

```@docs
tdot
dott
```

## Norm

The (2)-norm of a tensor is defined for a vector, second order tensor and fourth order tensor as

$\|\mathbf{a}\| = \sqrt{\mathbf{a} \cdot \mathbf{a}} \Leftrightarrow \|a_i\| = \sqrt{a_i a_i}$

$\|\mathbf{A}\| = \sqrt{\mathbf{A} : \mathbf{A}} \Leftrightarrow \|A_{ij}\| = \sqrt{A_{ij} A_{ij}}$

$\|\mathsf{A}\| = \sqrt{\mathsf{A} :: \mathsf{A}} \Leftrightarrow \|A_{ijkl}\| = \sqrt{A_{ijkl} A_{ijkl}}$

```@docs
norm
```

## Trace

The trace for a second order tensor is defined as the sum of the diagonal elements. This can be written as

$\text{tr}(\mathbf{A}) = \mathbf{I} : \mathbf{A} \Leftrightarrow \text{tr}(A_{ij}) = A_{ii}$

```@docs
trace
```

## Determinant

Determinant for a second order tensor.

```@docs
det
```

## Inverse

Inverse of a second order tensor such that

$\mathbf{A}^{-1} \cdot \mathbf{A} = \mathbf{I}$

where ``\mathbf{I}`` is the second order identitiy tensor.

```@docs
inv
```

## Transpose

Transpose of tensors is defined by changing the order of the tensor's "legs". The transpose of a vector/symmetric tensor is the vector/tensor itself. The transpose of a second order tensor can be written as:

$A_{ij}^\text{T} = A_{ji}$

and for a fourth order tensor the minor transpose can be written as

$A_{ijkl}^\text{t} = A_{jilk}$

and the major transpose as

$A_{ijkl}^\text{T} = A_{klij}$

```@docs
transpose
minortranspose
majortranspose
```

## Symmetric

The symmetric part of a second and fourth order tensor is defined by:

$\mathbf{A}^\text{sym} = \frac{1}{2}(\mathbf{A} + \mathbf{A}^\text{T}) \Leftrightarrow A_{ij}^\text{sym} = \frac{1}{2}(A_{ij} + A_{ji})$
$\mathsf{A}^\text{sym} = \frac{1}{2}(\mathsf{A} + \mathsf{A}^\text{t}) \Leftrightarrow A_{ijkl}^\text{sym} = \frac{1}{2}(A_{ijkl} + A_{jilk})$

```@docs
symmetric
minorsymmetric
majorsymmetric
```

## Skew symmetric

The skew symmetric part of a second order tensor is defined by

$\mathbf{A}^\text{skw} = \frac{1}{2}(\mathbf{A} - \mathbf{A}^\text{T}) \Leftrightarrow A^\text{skw}_{ij} = \frac{1}{2}(A_{ij} - A_{ji})$

The skew symmetric part of a symmetric tensor is zero.

```@docs
skew
```

## Deviator

The deviatoric part of a second order tensor is defined by

$\mathbf{A}^\text{dev} = \mathbf{A} - \frac{1}{3} \mathrm{trace}[\mathbf{A}] \mathbf{I} \Leftrightarrow A_{ij}^\text{dev} = A_{ij} - \frac{1}{3}A_{kk}\delta_{ij}$

```@docs
dev
```

## Cross product

The cross product between two vectors is defined as

$\mathbf{a} = \mathbf{b} \times \mathbf{c} \Leftrightarrow a_i = \epsilon_{ijk} b_j c_k$

```@docs
cross
```

## Eigenvalues and eigenvectors

The eigenvalues and eigenvectors of a (symmetric) second order tensor, ``\mathbf{A}`` can be solved from the eigenvalue problem

$\mathbf{A} \cdot \mathbf{v}_i = \lambda_i \mathbf{v}_i \qquad i = 1, \dots, \text{dim}$

where ``\lambda_i`` are the eigenvalues and ``\mathbf{v}_i`` are the corresponding eigenvectors.

```@docs
eig
```

## Rotations

```@docs
rotate
```

## Special operations

For computing a special dot product between two vectors $\mathbf{a}$ and $\mathbf{b}$ with a fourth order symmetric tensor $\mathbf{C}$ such that $a_k C_{ikjl} b_l$ there is `dotdot(a, C, b)`. This function is useful because it is the expression for the tangent matrix in continuum mechanics when the displacements are approximated by scalar shape functions.

```@docs
dotdot
```
