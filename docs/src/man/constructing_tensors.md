```@meta
DocTestSetup = quote
    srand(1234)
    using ContMechTensors
end
```

# Constructing tensors

Tensors can be created in multiple ways but they usually include `Tensor{order, dim}` or `SymmetricTensor{order, dim}`

```jldoctest
julia> zero(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.0
 0.0

julia> rand(Tensor{2, 3})
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> zero(SymmetricTensor{4, 2})
2×2×2×2 ContMechTensors.SymmetricTensor{4,2,Float64,9}:
[:, :, 1, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 1] =
 0.0  0.0
 0.0  0.0

[:, :, 1, 2] =
 0.0  0.0
 0.0  0.0

[:, :, 2, 2] =
 0.0  0.0
 0.0  0.0

julia> one(SymmetricTensor{2, 2})
2×2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 1.0  0.0
 0.0  1.0
```

Tensors can also be created by giving a tuple or an array with the same number of elements as the number of independent indices in the tensor:

```jldoctest
julia> Tensor{1,2}([1.0,2.0])
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 1.0
 2.0

julia> SymmetricTensor{2,2}((1.0,2.0,3.0))
2×2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 1.0  2.0
 2.0  3.0
```

It is also possible to create a tensor by giving a function `f(index...) -> v`:

```jldoctest
julia> SymmetricTensor{2,2}((i,j) -> i + j)
2×2 ContMechTensors.SymmetricTensor{2,2,Int64,3}:
 2  3
 3  4
```

A diagonal tensor can be created by either giving a number of a vector on the diagonal:

```jldoctest
julia> diagm(Tensor{2,2}, 2.0)
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 2.0  0.0
 0.0  2.0

julia> diagm(SymmetricTensor{2,3}, [1.0, 2.0, 3.0])
3×3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  3.0
```

## Converting to tensors

Sometimes it is necessary to convert between standard Julia `Array`'s and `Tensor`'s. This can be done
with `reinterpret`. For example, a `2×5` Julia `Array` can be translated to a vector of `Vec{2}` with the
following code (and then translated back again)

```jldoctest
julia> data = rand(2, 5)
2×5 Array{Float64,2}:
 0.590845  0.566237  0.794026  0.200586  0.246837
 0.766797  0.460085  0.854147  0.298614  0.579672

julia> tensor_data = reinterpret(Vec{2, Float64}, data, (5,))
5-element Array{ContMechTensors.Tensor{1,2,Float64,2},1}:
 [0.590845,0.766797]
 [0.566237,0.460085]
 [0.794026,0.854147]
 [0.200586,0.298614]
 [0.246837,0.579672]

julia> data = reinterpret(Float64, tensor_data, (2,5))
2×5 Array{Float64,2}:
 0.590845  0.566237  0.794026  0.200586  0.246837
 0.766797  0.460085  0.854147  0.298614  0.579672
```
