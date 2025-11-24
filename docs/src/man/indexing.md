```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
end
```

# Indexing

Indexing into a `(Symmetric)Tensor{dim, order}` is performed like for an `Array` of dimension `order`.

```jldoctest
julia> A = rand(Tensor{2, 2});

julia> A[1, 2]
0.21858665481883066

julia> B = rand(SymmetricTensor{4, 2});

julia> B[1, 2, 1, 2]
0.4942498668904206
```

Slicing will produce a `Tensor` of lower order.

```jldoctest
julia> A = rand(Tensor{2, 2});

julia> A[:, 1]
2-element Vec{2, Float64}:
 0.32597672886359486
 0.5490511363155669
```

Since `Tensor`s are immutable there is no `setindex!` function defined on them. Instead, use the functionality to create tensors from functions as described [here](@ref function_index). As an example, this sets the `[1,2]` index on a tensor to one and the rest to zero:

```jldoctest
julia> Tensor{2, 2}((i,j) -> i == 1 && j == 2 ? 1.0 : 0.0)
2×2 Tensor{2, 2, Float64, 4}:
 0.0  1.0
 0.0  0.0
```

For symmetric tensors, note that you should only set the lower triangular part of the tensor:

```jldoctest
julia> SymmetricTensor{2, 2}((i,j) -> i == 1 && j == 2 ? 1.0 : 0.0)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.0  0.0
 0.0  0.0

julia> SymmetricTensor{2, 2}((i,j) -> i == 2 && j == 1 ? 1.0 : 0.0)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.0  1.0
 1.0  0.0
```
