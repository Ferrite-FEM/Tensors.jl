# ContMechTensors

[![Build Status](https://travis-ci.org/KristofferC/ContMechTensors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/ContMechTensors.jl) [![codecov.io](https://codecov.io/github/KristofferC/ContMechTensors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/ContMechTensors.jl?branch=master)


## Creating Tensors

Tensors can be created in multiple ways but they usually include `Tensor{order, dim}` or `SymmetricTensor{order, dim}`

```jl
julia> Tensor{1, 2}()
2-element ContMechTensors.Tensor{1,2,Float64,1}:
 0.0
 0.0

julia> rand(Tensor{2, 3})
3x3 ContMechTensors.Tensor{2,3,Float64,1}:
 0.945867   0.312709  0.245964
 0.356161   0.726716  0.712027
 0.0946236  0.857122  0.386342

julia> zero(SymmetricTensor{4, 2})
2x2x2x2 ContMechTensors.SymmetricTensor{4,2,Float64,2}:
[:, :, 1, 1] =
 0.0  0.0
 0.0  0.0
 .
 .

julia> one(SymmetricTensor{2, 2})
2x2 ContMechTensors.SymmetricTensor{2,2,Float64,1}:
 1.0  0.0
 0.0  1.0
```

## Indexing

Indexing into a `(Symmetric)Tensor{dim, order}` is performed like for an `Array` of dimension `order`. The symbols `:x, :y, :z` can also be used.

```jl
julia> A = rand(Tensor{2, 2})
2x2 ContMechTensors.Tensor{2,2,Float64,1}:
 0.771481  0.865792
 0.860747  0.109753

julia> A[:x, :y]
0.8657915183351226

julia> A[1, 2]
0.8657915183351226

julia> B = rand(SymmetricTensor{4, 2});

julia> B[:x,:y,:x,:x]
0.10221501099081753
```

## Operations

### Double contractions

Double contractions of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n - 4`. The symbol `*` is overloaded for double contractions but `dcontract(A, B)` can also be used. To store the result in a preallocated tensor, use `dcontract!(C, A, B)`

```jl
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> A*B
0.9392510193487607

julia> dcontract(A,B)
0.9392510193487607
```

### Single contraction (dot products)

Single contractions or scalar products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n - 2`. The symbol `⋅` is overloaded for scalar products but `dot(A, B)` can also be used. To store the result in a preallocated tensor, use `otimes!(C, A, B)`. Since dot products between symmetric tensors does not give a symmetric result it is not implemented.

```jl
julia> A = rand(Tensor{2, 2})
2x2 ContMechTensors.Tensor{2,2,Float64,1}:
 0.246704  0.379757
 0.180964  0.947665

julia> B = rand(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,1}:
 0.772635
 0.0625623

julia> A ⋅ B
2-element ContMechTensors.Tensor{1,2,Float64,1}:
 0.214371
 0.199108
```

### Tensor products

Tensor products or open products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n`.
The symbol `⊗` is overloaded for tensor products but `otimes(a, b)` can also be used. To store the result in a preallocated tensor, use `otimes!(c, a, b)`

```jl
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> A ⊗ B
2x2x2x2 ContMechTensors.SymmetricTensor{4,2,Float64,2}:
[:, :, 1, 1] =
 0.219546  0.874947
 0.874947  0.226921

[:, :, 2, 1] =
 0.022037   0.0878232
 0.0878232  0.0227773
.
.
```

### Other operators:

For all type of tensors the following operators are implemented; `trace`, `norm`.

For second order tensors: `dev`, `det`.

For second and fourth order tensors: `inv`, `transpose`.



