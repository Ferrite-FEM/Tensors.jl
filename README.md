# ContMechTensors

[![Build Status](https://travis-ci.org/KristofferC/ContMechTensors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/ContMechTensors.jl)

This Julia package provides fast operations with symmetric/unsymmetric tensors of order 1, 2 and 4. The tensors are stack allocated which means that there is no need to preallocate results of operations and nice infix notation can be used without a performance penalty. For the symmetric tensors, when possible, the symmetry is exploited for better performance.

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

Tensors can also be created by giving a tuple or an array with the same number of elements as the number of independent indices in the tensor:

```jl
julia> Tensor{1,2}([1.0,2.0])
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 1.0
 2.0

julia> SymmetricTensor{2,2}((1.0,2.0,3.0))
2x2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 1.0  2.0
 2.0  3.0
```

It is also possible to create a tensor by giving a function `f(index...) -> v`:

```jl
julia> SymmetricTensor{2,2}((i,j) -> i + j)
2x2 ContMechTensors.SymmetricTensor{2,2,Int64,3}:
 2  3
 3  4
```

A diagonal tensor can be created by either giving a number of a vector on the diagonal:

```jl
julia> diagm(Tensor{2,2}, 2.0)
2x2 ContMechTensors.Tensor{2,2,Float64,4}:
 2.0  0.0
 0.0  2.0

julia> diagm(SymmetricTensor{2,3}, [1.0, 2.0, 3.0])
3x3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  3.0
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

```jl
julia> a = rand(Vec{2});

julia> setindex(a, 1.337, 2)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.026256
 1.337
```


## Operations

### Single contraction (dot product)

Single contractions or scalar products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n - 2`. The symbol `⋅`, written `\cdot`, is overloaded for single contraction.

```jl
julia> A = rand(Tensor{2, 2})
2x2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.0928652  0.664058
 0.799669   0.979861

julia> B = rand(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.687288
 0.461646

julia> dot(A, B)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.370385
 1.00195 

julia> A ⋅ B
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.370385
 1.00195
```


### Double contraction

Double contractions contracts the two most inner "legs" of the tensors. The result of a double contraction between a tensor of order `n` and a tensor with order `m` gives a tensor with order `m + n - 4`. The symbol `⊡`, written `\boxdot`, is overloaded for double contraction. The reason `:` is not used is because it does not have the same precedence as multiplication.

```jl
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
0.9392510193487607

julia> A ⊡ B
0.9392510193487607
```


### Tensor product (open product)

Tensor products or open product of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n`. The symbol `⊗`, written `\otimes`, is overloaded for tensor products.

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

For vectors (first order tensors): `norm`

For second order tensors: `norm`, `trace` (`vol`), `det`, `inv`, `transpose`, `symmetric`, `skew`, `eig`, `mean` defined as `trace(s) / 3`, and `dev` defined as `s - mean(s) * I`.

For fourth order tensors: `norm` and `trace`

There is also a special function for computing `F' ⋅ F` between two general second order tensors which is called `tdot` and returns a `SymmetricTensor`.

### Storing tensors in `type`s.

Even though a user mostly deals with the `Tensor{order, dim, T}` parameters, the full parameter list for a tensor is actually `Tensor{order, dim, T, N}` where `N` is the number of independent elements in the tensor. The reason for this is that the internal storage is a `NTuple{N, T}`. In order to get good performance when storing tensors in other types it is importatant that the container type is also parametrized on `N`. For example, when storing one symmetric second order tensor and one unsymmetric tensor, this is the preferred way:

```jl
immutable Container{dim, T, N, M}
    sym_tens::SymmetricTensor{2, dim, T, N}
    tens::Tensor{2, dim, T, M}
end
```

Leaving out the `M` and `N` would lead to bad performance.
