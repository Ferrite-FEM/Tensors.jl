```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
end
```

# Binary Operations

```@index
Pages = ["binary_operators.md"]
```

## Dot product (single contraction)

The dot product (or single contraction) between a tensor of order `n` and a tensor of order `m` is a tensor of order `m + n - 2`. For example, single contraction between two vectors ``\mathbf{b}`` and ``\mathbf{c}`` can be written as:

```math
a = \mathbf{b} \cdot \mathbf{c} \Leftrightarrow a = b_i c_i
```

and single contraction between a second order tensor ``\mathbf{B}`` and a vector ``\mathbf{c}``:

```math
\mathbf{a} = \mathbf{B} \cdot \mathbf{c} \Leftrightarrow a_i = B_{ij} c_j
```

```@docs
dot
```

## Double contraction

A double contraction between two tensors contracts the two most inner indices. The result of a double contraction between a tensor of order `n` and a tensor of order `m` is a tensor of order `m + n - 4`. For example, double contraction between two second order tensors ``\mathbf{B}`` and ``\mathbf{C}`` can be written as:

```math
a = \mathbf{B} : \mathbf{C} \Leftrightarrow a = B_{ij} C_{ij}
```

and double contraction between a fourth order tensor ``\mathsf{B}`` and a second order tensor ``\mathbf{C}``:

```math
\mathbf{A} = \mathsf{B} : \mathbf{C} \Leftrightarrow A_{ij} = B_{ijkl} C_{kl}
```

```@docs
dcontract
```

## Tensor product (open product)

The tensor product (or open product) between a tensor of order `n` and a tensor of order `m` is a tensor of order `m + n`.  For example, open product between two vectors ``\mathbf{b}`` and ``\mathbf{c}`` can be written as:

```math
\mathbf{A} = \mathbf{b} \otimes \mathbf{c} \Leftrightarrow A_{ij} = b_i c_j
```

and open product between two second order tensors ``\mathbf{B}`` and ``\mathbf{C}``:

```math
\mathsf{A} = \mathbf{B} \otimes \mathbf{C} \Leftrightarrow A_{ijkl} = B_{ij} C_{kl}
```

```@docs
otimes
```

### Permuted tensor products

Two commonly used permutations of the open product are the "upper" open product
(``\bar{\otimes}``) and "lower" open product (``\underline{\otimes}``) defined
between second order tensors ``\mathbf{B}`` and ``\mathbf{C}`` as

```math
\mathsf{A} = \mathbf{B} \bar{\otimes} \mathbf{C} \Leftrightarrow A_{ijkl} = B_{ik} C_{jl}\\
\mathsf{A} = \mathbf{B} \underline{\otimes} \mathbf{C} \Leftrightarrow A_{ijkl} = B_{il} C_{jk}
```

```@docs
otimesu
otimesl
```
