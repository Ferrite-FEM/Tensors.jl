# Constructing tensors

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
