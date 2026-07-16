module Tensors

import Statistics
using Statistics: mean
using LinearAlgebra
using StaticArrays
import ForwardDiff

# re-exports from LinearAlgebra
export ⋅, ×, dot, diagm, tr, det, norm, eigvals, eigvecs, eigen
# re-exports from Statistics
export mean

export AbstractTensor, SymmetricTensor, Tensor, MixedTensor
export Vec, FourthOrderTensor, SecondOrderTensor
export MixedTensor2, MixedTensor3, MixedTensor4

export otimes, ⊗, ⊡, dcontract, dev, vol, symmetric, skew, minorsymmetric, majorsymmetric
export otimesu, otimesl
export minortranspose, majortranspose, isminorsymmetric, ismajorsymmetric
export tdot, dott, dotdot
export hessian, gradient, curl, divergence, laplace
export @implement_gradient
export basevec, eᵢ
export rotate, rotation_tensor
export tovoigt, tovoigt!, fromvoigt, tomandel, tomandel!, frommandel

#########
# Types #
#########
abstract type AbstractTensor{order, dim, T <: Number} <: AbstractArray{T, order} end

"""
    SymmetricTensor{order,dim,T<:Number}

Symmetric tensor type supported for `order ∈ (2,4)` and `dim ∈ (1,2,3)`.
`SymmetricTensor{4}` is a minor symmetric tensor, such that
`A[i,j,k,l] == A[j,i,k,l]` and `A[i,j,k,l] == A[i,j,l,k]`.

# Examples
```jldoctest
julia> SymmetricTensor{2,2,Float64}((1.0, 2.0, 3.0))
2×2 SymmetricTensor{2, 2, Float64, 3}:
 1.0  2.0
 2.0  3.0
```
"""
struct SymmetricTensor{order, dim, T, M} <: AbstractTensor{order, dim, T}
    data::NTuple{M, T}
end

"""
    Tensor{order,dim,T<:Number}

Tensor type supported for `order ∈ (1,2,3,4)` and `dim ∈ (1,2,3)`.

# Examples
```jldoctest
julia> Tensor{1,3,Float64}((1.0, 2.0, 3.0))
3-element Vec{3, Float64}:
 1.0
 2.0
 3.0
```
"""
struct Tensor{order, dim, T, M} <: AbstractTensor{order, dim, T}
    data::NTuple{M, T}
end

"""
    MixedTensor{order, dims <: Tuple, T<:Number}

`MixedTensor` have different dimensions for each basis, described by the `dims`
tuple type, e.g. `dims = Tuple{2, 1, 3}` for a 3rd order tensor with size `(2, 1, 3)`.
It supports `order ∈ (1,2,3,4)` and `dim ∈ (1,2,3)`

The following aliases can simplify construction and dispatch:

| Order | Alias | Size |
| :-- | :-- | :-- |
| 2nd | `MixedTensor2{d1, d2, T}` | `(d1, d2)` |
| 3rd | `MixedTensor3{d1, d2, d3, T}` | `(d1, d2, d3)` |
| 4th | `MixedTensor4{d1, d2, d3, d4, T}` | `(d1, d2, d3, d4)` |

# Examples
```jldoctest
julia> MixedTensor{2, Tuple{2, 3}, Float64}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
2×3 MixedTensor2{2, 3, Float64, 6}:
 1.0  3.0  5.0
 2.0  4.0  6.0
```
Or, shorter using the `MixedTensor2` alias
```jldoctest
julia> a = MixedTensor2{2, 3}((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
2×3 MixedTensor2{2, 3, Float64, 6}:
 1.0  3.0  5.0
 2.0  4.0  6.0

julia> a ⋅ a'
2×2 Tensor{2, 2, Float64, 4}:
 35.0  44.0
 44.0  56.0
```

!!! note
    After performing operations with `MixedTensor`s, they become regular `Tensor`s
    whenever possible, i.e. when all dimensions are the same.
    This is exemplified above with the dot-product, `a ⋅ a'`, but applies to all operations.

"""
struct MixedTensor{order, dims, T, M} <: AbstractTensor{order, dims, T}
    data::NTuple{M, T}
end

###############
# Typealiases #
###############
const Vec{dim, T, M} = Tensor{1, dim, T, dim}

const MixedTensor2{d1, d2, T, M} = MixedTensor{2, Tuple{d1, d2}, T, M}
const MixedTensor3{d1, d2, d3, T, M} = MixedTensor{3, Tuple{d1, d2, d3}, T, M}
const MixedTensor4{d1, d2, d3, d4, T, M} = MixedTensor{4, Tuple{d1, d2, d3, d4}, T, M}

const AllTensors{dim, T} = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                 SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                 Vec{dim, T}, Tensor{3, dim, T}}

const SecondOrderTensor{dim, T}   = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}, MixedTensor{2, dim, T}}
const FourthOrderTensor{dim, T}   = Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}, MixedTensor{4, dim, T}}
const SymmetricTensors{dim, T}    = Union{SymmetricTensor{2, dim, T}, SymmetricTensor{4, dim, T}}
const NonSymmetricTensors{dim, T} = Union{Tensor{2, dim, T}, Tensor{4, dim, T}, Vec{dim, T}}

##############################
# Utility/Accessor Functions #
##############################
import Base.@pure

@inline get_data(t::AbstractTensor) = t.data

@pure n_components(::Type{SymmetricTensor{2, dim}}) where {dim} = dim * dim - div((dim - 1) * dim, 2)
@pure function n_components(::Type{SymmetricTensor{4, dim}}) where {dim}
    n = n_components(SymmetricTensor{2, dim})
    return n * n
end
@pure n_components(::Type{Tensor{order, dim}}) where {order, dim} = dim^order
@pure n_components(::Type{MixedTensor{order, dims}}) where {order, dims} = *(size(MixedTensor{order, dims})...)

if isdefined(Core, :TypeEgal)
    get_type(T::Union{Core.TypeEq, Core.TypeEgal}) = Base.type_parameter(T)
else
    @pure get_type(::Type{Type{X}}) where {X} = X
end

@pure get_base(::Type{<:Tensor{order, dim}})          where {order, dim}  = Tensor{order, dim}
@pure get_base(::Type{<:SymmetricTensor{order, dim}}) where {order, dim}  = SymmetricTensor{order, dim}
@pure get_base(::Type{<:MixedTensor{order, dims}})    where {order, dims} = MixedTensor{order, dims}

@pure Base.eltype(::Type{Tensor{order, dim, T, M}})          where {order, dim, T, M} = T
@pure Base.eltype(::Type{Tensor{order, dim, T}})             where {order, dim, T}    = T
@pure Base.eltype(::Type{Tensor{order, dim}})                where {order, dim}       = Any
@pure Base.eltype(::Type{SymmetricTensor{order, dim, T, M}}) where {order, dim, T, M} = T
@pure Base.eltype(::Type{SymmetricTensor{order, dim, T}})    where {order, dim, T}    = T
@pure Base.eltype(::Type{SymmetricTensor{order, dim}})       where {order, dim}       = Any
@pure Base.eltype(::Type{MixedTensor{order, dims, T, M}})    where {order, dims, T, M} = T
@pure Base.eltype(::Type{MixedTensor{order, dims, T}})       where {order, dims, T}    = T
@pure Base.eltype(::Type{MixedTensor{order, dims}})          where {order, dims}       = Any

############################
# Abstract Array interface #
############################
Base.IndexStyle(::Type{<:SymmetricTensor}) = IndexCartesian()
Base.IndexStyle(::Type{<:Tensor}) = IndexLinear()
Base.IndexStyle(::Type{<:MixedTensor}) = IndexLinear()

########
# Size #
########
Base.size(::TT) where {TT <: AbstractTensor} = size(TT)
Base.size(::Type{<:Vec{dim}})               where {dim} = (dim,)
Base.size(::Type{<:SecondOrderTensor{dim}}) where {dim} = (dim, dim)
Base.size(::Type{<:Tensor{3, dim}})         where {dim} = (dim, dim, dim)
Base.size(::Type{<:FourthOrderTensor{dim}}) where {dim} = (dim, dim, dim, dim)
Base.size(::Type{<:MixedTensor{1, Tuple{d1}}}) where {d1} = (d1,)
Base.size(::Type{<:MixedTensor2{d1, d2}}) where {d1, d2} = (d1, d2)
Base.size(::Type{<:MixedTensor3{d1, d2, d3}}) where {d1, d2, d3} = (d1, d2, d3)
Base.size(::Type{<:MixedTensor4{d1, d2, d3, d4}}) where {d1, d2, d3, d4} = (d1, d2, d3, d4)

# Also define length for the type itself
Base.length(::Type{Tensor{order, dim, T, M}}) where {order, dim, T, M} = M

include("indexing.jl")
include("einsum.jl")
include("simd_lowering.jl")
include("maps.jl")
include("mixed_tensors.jl")
include("tensor_ops_errors.jl")
include("constructors.jl")
include("promotion_conversion.jl")
include("basic_operations.jl")
include("tensor_products.jl")
include("transpose.jl")
include("symmetric.jl")
include("math_ops.jl")
include("eigen.jl")
include("special_ops.jl")
include("automatic_differentiation.jl")
include("voigt.jl")
include("precompile.jl")

end # module
