```@meta
DocTestSetup = quote
    srand(1234)
    using ContMechTensors
end
```

# Binary Operations

## Single contraction (dot product)

Single contractions or scalar products of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n - 2`. The symbol `⋅`, written `\cdot`, is overloaded for single contraction.

```jldoctest
julia> A = rand(Tensor{2, 2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> B = rand(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.794026
 0.854147

julia> dot(A, B)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.952796
 1.00184

julia> A ⋅ B
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.952796
 1.00184
```


## Double contraction

Double contractions contracts the two most inner "legs" of the tensors. The result of a double contraction between a tensor of order `n` and a tensor with order `m` gives a tensor with order `m + n - 4`. The symbol `⊡`, written `\boxdot`, is overloaded for double contraction. The reason `:` is not used is because it does not have the same precedence as multiplication.

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
1.9732018397544984

julia> A ⊡ B
1.9732018397544984
```


## Tensor product (open product)

Tensor products or open product of a tensor with order `n` and a tensor with order `m` gives a tensor with order `m + n`. The symbol `⊗`, written `\otimes`, is overloaded for tensor products.

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> A ⊗ B
2×2×2×2 ContMechTensors.SymmetricTensor{4,2,Float64,9}:
[:, :, 1, 1] =
 0.271839  0.352792
 0.352792  0.260518

[:, :, 2, 1] =
 0.469146  0.608857
 0.608857  0.449607

[:, :, 1, 2] =
 0.469146  0.608857
 0.608857  0.449607

[:, :, 2, 2] =
 0.504668  0.654957
 0.654957  0.48365
```
