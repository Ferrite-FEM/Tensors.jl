# ContMechTensors

[![Build Status](https://travis-ci.org/KristofferC/ContMechTensors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/ContMechTensors.jl) 

This Julia package provides fast operations with symmetric/unsymmetric tensors of order 1, 2 and 4. The tensors are stack allocated which means that there is no need to preallocate results of operations and nice infix notation can be used without a perfomance penalty. For the symmetric tensors, when possible, the symmetriy is exploited for better performance.

Note that this package might not provide satisfactory performance on julia v0.4 because julia v0.4 lacks various optimizations to tuples that julia v0.5 has.

## Creating Tensors

Tensors can be created in multiple ways but they usually include `Tensor{order, dim}` or `SymmetricTensor{order, dim}`

```jl
julia> zero(Tensor{1, 2})
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

Indexing into a `(Symmetric)Tensor{dim, order}` is performed like for an `Array` of dimension `order`. 

```jl
julia> A = rand(Tensor{2, 2});

julia> A[1, 2]
0.8657915183351226

julia> B = rand(SymmetricTensor{4, 2});

julia> B[1, 2, 1, 2]
0.10221501099081753
```

In order to set an index the function `setindex(t, value, index...)` is used. This returns a new tensor with the modified index. Explicitly setting indicies is not recommended in performance critical code since it will invoke dynamic dispatch. It is provided as a means of convenience when working in for example the REPL.

```julia
julia> a = rand(Vec{2});

julia> setindex(a, 1.337, 2)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.026256
 1.337   
```


## Operations

The symbol `*` is overloaded for double contractions between fourth and second order tensors, and single contractions between two second order tensors or between a second and first order tensor.

### Double contractions

Double contractions contracts the two most inner "legs" of the tensors. The result of a double contraction between a tensor of order `n` and a tensor with order `m` gives a tensor with order `m + n - 4`. The symbol `⊡`, written `\boxdot` is overloaded for double contraction. The reason `:` is not used is because it does not have the same precedence as multiplication.

```jl
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
0.9392510193487607

julia> A ⊡ B
0.9392510193487607
```

### Single contraction (dot products)

Single contractions or scalar products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n - 2`. The symbol `⋅` is overloaded for single contractiom.

```jl
julia> A = rand(Tensor{2, 2})
2x2 ContMechTensors.Tensor{2,2,Float64,1}:
 0.246704  0.379757
 0.180964  0.947665

julia> B = rand(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,1}:
 0.772635
 0.0625623

julia> dot(A, B)
2-element ContMechTensors.Tensor{1,2,Float64,1}:
 0.214371
 0.199108
```

### Tensor products

Tensor products or open products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n`T. he symbol `⊗` is overloaded for tensor products.

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

For second order tensors: `dev`, `det`, `inv`, `transpose`.

There is also a special function for computing `F' ⋅ F` between two general second order tensors which is called `tdot` and returns a `SymmetricTensor`.



