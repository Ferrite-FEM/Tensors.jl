```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
end
```

# Constructing tensors

Tensors can be created in multiple ways but they usually include running a function on tensor types of which there are two kinds, `Tensor{order, dim, T}` for non-symmetric tensors and `SymmetricTensor{order, dim, T}` for symmetric tensors.
The parameter `order` is an integer of value 1, 2 or 4, excluding 1 for symmetric tensors. The second parameter `dim` is an integer which corresponds to the dimension of the tensor and can be 1, 2 or 3. The last parameter `T` is the number type that the tensors contain, i.e. `Float64` or `Float32`.

```@docs
Tensors.Tensor
Tensors.SymmetricTensor
```

## [Zero tensors](@id zero_tensors)

A tensor with only zeros is created using the function `zero`, applied to the type of tensor that should be created:

```jldoctest
julia> zero(Tensor{1, 2})
2-element Tensor{1,2,Float64,2}:
 0.0
 0.0
```

By default, a tensor of `Float64`s is created, but by explicitly giving the `T` parameter this can be changed:

```jldoctest
julia> zero(SymmetricTensor{4, 2, Float32})
2×2×2×2 SymmetricTensor{4,2,Float32,9}:
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
```

A Julia `Array` with zeroed tensors can be created with `zeros`, with the tensor type and dimensions of the array as arguments:

```jldoctest
julia> zeros(Tensor{2,2}, 2, 3)
2×3 Array{Tensor{2,2,Float64,4},2}:
 [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]
 [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]  [0.0 0.0; 0.0 0.0]
```

## Constant tensors

A tensor filled with ones is created using the function `ones`, applied to the type of tensor that should be created:

```jldoctest
julia> ones(Tensor{2,2})
2×2 Tensor{2,2,Float64,4}:
 1.0  1.0
 1.0  1.0
```

By default, a tensor of `Float64`s is created, but by explicitly giving the `T` parameter this can be changed, like for [zero](@ref zero_tensors).

!!! note
    The function `ones` has double meaning: it can create a tensor filled with ones
    (as described above) or create a Julia `Array` with [identity tensors](@ref identity_tensors).
    Thus, to create an `Array` with tensors filled with ones, instead use array comprehension:

    ```jldoctest
    julia> [ones(Tensor{2,2}) for i in 1:2, j in 1:3]
    2×3 Array{Tensor{2,2,Float64,4},2}:
     [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]
     [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]  [1.0 1.0; 1.0 1.0]
    ```


## Random tensors

A tensor with random numbers is created using the function `rand`, applied to the type of tensor that should be created:

```jldoctest
julia> rand(Tensor{2, 3})
3×3 Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837
```

By specifying the type, `T`, a tensor of different type can be obtained:

```jldoctest
julia> rand(SymmetricTensor{2,3,Float32})
3×3 SymmetricTensor{2,3,Float32,6}:
 0.0107703  0.305865  0.2082
 0.305865   0.405684  0.257278
 0.2082     0.257278  0.958491
```

## [Identity tensors](@id identity_tensors)

Identity tensors can be created for orders 2 and 4. The components of the second order identity tensor ``\mathbf{I}`` are defined as ``I_{ij} = \delta_{ij}``, where ``\delta_{ij}`` is the Kronecker delta. The fourth order identity tensor ``\mathsf{I}`` is the resulting tensor from taking the derivative of a second order tensor ``\mathbf{A}`` with itself:

```math
\mathsf{I} = \frac{\partial \mathbf{A}}{\partial \mathbf{A}} \Leftrightarrow I_{ijkl} = \frac{\partial A_{ij}}{\partial A_{kl}} = \delta_{ik} \delta_{jl}.
```

The symmetric fourth order tensor, ``\mathsf{I}^\text{sym}``, is the resulting tensor from taking the derivative of a symmetric second order tensor ``\mathbf{A}^\text{sym}`` with itself:

```math
\mathsf{I}^\text{sym} = \frac{\partial \mathbf{A}^\text{sym}}{\partial \mathbf{A}^\text{sym}} \Leftrightarrow I^\text{sym}_{ijkl} = \frac{\partial A^\text{sym}_{ij}}{\partial A^\text{sym}_{kl}} = \frac{1}{2} (\delta_{ik} \delta_{jl} + \delta_{il} \delta_{jk}).
```

Identity tensors are created using the function `one`, applied to the type of tensor that should be created:

```jldoctest
julia> one(SymmetricTensor{2, 2})
2×2 SymmetricTensor{2,2,Float64,3}:
 1.0  0.0
 0.0  1.0
```

A Julia `Array` with identity tensors can be created with `ones`, with the tensor type and dimensions of the array as arguments:

```jldoctest
julia> ones(Tensor{2,2}, 2, 2)
2×2 Array{Tensor{2,2,Float64,4},2}:
 [1.0 0.0; 0.0 1.0]  [1.0 0.0; 0.0 1.0]
 [1.0 0.0; 0.0 1.0]  [1.0 0.0; 0.0 1.0]
```

## Base vectors

```@docs
basevec
```

## From arrays / tuples

Tensors can also be created from a tuple or an array with the same number of elements as the number of independent indices in the tensor. For example, a first order tensor (vector) in two dimensions is here created from a vector of length two:

```jldoctest
julia> Tensor{1,2}([1.0,2.0])
2-element Tensor{1,2,Float64,2}:
 1.0
 2.0
```

Below, a second order symmetric tensor in two dimensions is created from a tuple. Since the number of independent indices in this tensor is three, the length of the tuple is also three. For symmetric tensors, the order of the numbers in the input tuple is column by column, starting at the diagonal.

```jldoctest
julia> SymmetricTensor{2,2}((1.0,2.0,3.0))
2×2 SymmetricTensor{2,2,Float64,3}:
 1.0  2.0
 2.0  3.0
```

## [From a function](@id function_index)

A tensor can be created from a function `f(indices...) -> v` which maps a set of indices to a value. The number of arguments of the function should be equal to the order of the tensor.

```jldoctest
julia> SymmetricTensor{2,2,Float64}((i,j) -> i + j)
2×2 SymmetricTensor{2,2,Float64,3}:
 2.0  3.0
 3.0  4.0
```

For symmetric tensors, the function is only called for the lower triangular part.

## Diagonal tensors

A diagonal second order tensor can be created by either giving a number or a vector that should appear on the diagonal:

```jldoctest
julia> diagm(Tensor{2,2}, 2.0)
2×2 Tensor{2,2,Float64,4}:
 2.0  0.0
 0.0  2.0

julia> diagm(SymmetricTensor{2,3}, [1.0, 2.0, 3.0])
3×3 SymmetricTensor{2,3,Float64,6}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  3.0
```

## Converting to tensors

Sometimes it is necessary to convert between standard Julia `Array`'s and `Tensor`'s. When the number type is a bits type (like for floats or integers) this is conveniently done by the `reinterpret` function. For example, a `2×5` Julia `Array` can be translated to a vector of `Vec{2}` with the
following code

```jldoctest fromarray
julia> data = rand(2, 5)
2×5 Array{Float64,2}:
 0.590845  0.566237  0.794026  0.200586  0.246837
 0.766797  0.460085  0.854147  0.298614  0.579672

julia> tensor_data = reinterpret(Vec{2, Float64}, vec(data))
5-element reinterpret(Tensor{1,2,Float64,2}, ::Array{Float64,1}):
 [0.5908446386657102, 0.7667970365022592]
 [0.5662374165061859, 0.4600853424625171]
 [0.7940257103317943, 0.8541465903790502]
 [0.20058603493384108, 0.2986142783434118]
 [0.24683718661000897, 0.5796722333690416]
```

The data can also be reinterpreted back to a Julia `Array`

```jldoctest fromarray
julia> data = reshape(reinterpret(Float64, tensor_data), (2, 5))
2×5 reshape(reinterpret(Float64, reinterpret(Tensor{1,2,Float64,2}, ::Array{Float64,1})), 2, 5) with eltype Float64:
 0.590845  0.566237  0.794026  0.200586  0.246837
 0.766797  0.460085  0.854147  0.298614  0.579672
```
