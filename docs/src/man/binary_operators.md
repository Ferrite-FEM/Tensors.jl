# Binary Operations

## Single contraction (dot product)

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


## Double contraction

Double contractions contracts the two most inner "legs" of the tensors. The result of a double contraction between a tensor of order `n` and a tensor with order `m` gives a tensor with order `m + n - 4`. The symbol `⊡`, written `\boxdot`, is overloaded for double contraction. The reason `:` is not used is because it does not have the same precedence as multiplication.

```jl
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
0.9392510193487607

julia> A ⊡ B
0.9392510193487607
```


## Tensor product (open product)

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