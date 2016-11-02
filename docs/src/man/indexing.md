# Indexing

Indexing into a `(Symmetric)Tensor{dim, order}` is performed like for an `Array` of dimension `order`.

```jl
julia> A = rand(Tensor{2, 2});

julia> A[1, 2]
0.8657915183351226

julia> B = rand(SymmetricTensor{4, 2});

julia> B[1, 2, 1, 2]
0.10221501099081753
```

In order to set an index the function `setindex(t, value, index...)` is used. This returns a new tensor with the modified index. Explicitly setting indices is not recommended in performance critical code since it will invoke dynamic dispatch. It is provided as a means of convenience when working in for example the REPL.

```jl
julia> a = rand(Vec{2});

julia> setindex(a, 1.337, 2)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.026256
 1.337
```
