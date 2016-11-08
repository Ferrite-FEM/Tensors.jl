```@meta
DocTestSetup = quote
    srand(1234)
    using ContMechTensors
end
```

# Binary Operations

```@index
Pages = ["binary_operators.md"]
```

## Dot product (single contraction)

Dot product or single contraction of a tensor of order `n` and a tensor of order `m` gives a tensor of order `m + n - 2`. For example, single contraction between two vectors ``\mathbf{b}`` and ``\mathbf{c}`` can be written as:

$a = \mathbf{b} \cdot \mathbf{c} \Leftrightarrow a = b_i c_i$

and single contraction between a second order tensor ``\mathbf{B}`` and a vector ``\mathbf{c}``:

$\mathbf{a} = \mathbf{B} \cdot \mathbf{c} \Leftrightarrow a_i = B_{ij} c_j$

```@docs
dot
```

## Double contraction

Double contractions contracts the two most inner "legs" of the tensors. The result of a double contraction between a tensor of order `n` and a tensor of order `m` gives a tensor of order `m + n - 4`. For example, double contraction between two second order tensors ``\mathbf{B}`` and ``\mathbf{C}`` can be written as:

$a = \mathbf{B} : \mathbf{C} \Leftrightarrow a = B_{ij} C_{ij}$

and double contraction between a fourth order tensor ``\mathsf{B}`` and a second order tensor ``\mathbf{C}``:

$\mathbf{A} = \mathsf{B} : \mathbf{C} \Leftrightarrow A_{ij} = B_{ijkl} C_{kl}$

```@docs
dcontract
```

## Tensor product (open product)

Tensor products or open product of a tensor of order `n` and a tensor of order `m` gives a tensor of order `m + n`.  For example, open product between two vectors ``\mathbf{b}`` and ``\mathbf{c}`` can be written as:

$\mathbf{A} = \mathbf{b} \otimes \mathbf{c} \Leftrightarrow A_{ij} = b_i c_j$

and open product between two second order tensors ``\mathbf{B}`` and ``\mathbf{C}``:

$\mathsf{A} = \mathbf{B} \otimes \mathbf{C} \Leftrightarrow A_{ijkl} = B_{ij} C_{kl}$

```@docs
otimes
```
