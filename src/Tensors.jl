module Tensors

import Base.@pure

import Statistics
using Statistics: mean
using LinearAlgebra
using StaticArrays
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
2×2 SymmetricTensor{2, 2, Float64}:
 1.0  2.0
 2.0  3.0
```
"""
struct SymmetricTensor{order, dim, T} <: AbstractTensor{order, dim, T}
    data::NTuple{div(dim*(dim+1),2)^div(order,2), T}
end

"""
    Tensor{order,dim,T<:Number}

Tensor type supported for `order ∈ (1,2,4)` and `dim ∈ (1,2,3)`.

# Examples
```jldoctest
julia> Tensor{1,3,Float64}((1.0, 2.0, 3.0))
3-element Vec{3, Float64}:
 1.0
 2.0
 3.0
```
"""
struct Tensor{order, dim, T} <: AbstractTensor{order, dim, T}
    data::NTuple{dim^order, T}
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
2×2 Tensor{2, 2, Float64}:
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
    MixedTensor{order, dims, T, M}(data::NTuple) where {order, dims <: Tuple, T, M} = new{order, dims, T, M}(data)
end
const MixedTensor2{d1, d2, T, M} = MixedTensor{2, Tuple{d1, d2}, T, M}
const MixedTensor3{d1, d2, d3, T, M} = MixedTensor{3, Tuple{d1, d2, d3}, T, M}
const MixedTensor4{d1, d2, d3, d4, T, M} = MixedTensor{4, Tuple{d1, d2, d3, d4}, T, M}

# The structural invariants -- one dimension per index, component count
# matching -- are enforced here since they cannot live on the struct itself.
@inline function _check_mixed_parameters(::Type{MixedTensor{order, dims}}, M::Int) where {order, dims <: Tuple}
    n = length(dims.parameters)
    n == order || throw(ArgumentError("MixedTensor{$order, $dims}: $n dimensions given for order $order"))
    N = n_components(MixedTensor{order, dims})
    M == N || throw(ArgumentError("MixedTensor{$order, $dims}: size requires $N components, got $M"))
    return nothing
end
@inline function MixedTensor{order, dims}(data::NTuple{M, T}) where {order, dims <: Tuple, T, M}
    _check_mixed_parameters(MixedTensor{order, dims}, M)
    return MixedTensor{order, dims, T, M}(data)
end
@inline function MixedTensor{order, dims, T}(data::NTuple{M, T2}) where {order, dims <: Tuple, T, T2, M}
    _check_mixed_parameters(MixedTensor{order, dims}, M)
    return MixedTensor{order, dims, T, M}(data)
end
MixedTensor{order, dims}(data::Tuple{Vararg{Any, M}}) where {order, dims, M} = MixedTensor{order, dims}(promote(data...))

###############
# Typealiases #
###############
const Vec{dim, T} = Tensor{1, dim, T}

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
get_data(t::AbstractTensor) = t.data

@pure n_components(::Type{SymmetricTensor{2, dim}}) where {dim} = dim*dim - div((dim-1)*dim, 2)
@pure function n_components(::Type{SymmetricTensor{4, dim}}) where {dim}
    n = n_components(SymmetricTensor{2, dim})
    return n*n
end
@pure n_components(::Type{Tensor{order, dim}}) where {order, dim} = dim^order

# Steal base implementation of "prod" to safely mark with @pure 
@pure n_components(::Type{MixedTensor{order, dims}}) where {order, dims} = *(size(MixedTensor{order, dims})...)

if isdefined(Core, :TypeEgal)
    get_type(T::Union{Core.TypeEq, Core.TypeEgal}) = Base.type_parameter(T)
else
    @pure get_type(::Type{Type{X}}) where {X} = X
end

@pure get_base(::Type{<:Tensor{order, dim}})          where {order, dim} = Tensor{order, dim}
@pure get_base(::Type{<:SymmetricTensor{order, dim}}) where {order, dim} = SymmetricTensor{order, dim}
@pure get_base(::Type{<:MixedTensor{order, dims}})    where {order, dims} = MixedTensor{order, dims}

@pure Base.eltype(::Type{Tensor{order, dim, T}})             where {order, dim, T}    = T
@pure Base.eltype(::Type{Tensor{order, dim}})                where {order, dim}       = Any
@pure Base.eltype(::Type{SymmetricTensor{order, dim, T}})    where {order, dim, T}    = T
@pure Base.eltype(::Type{SymmetricTensor{order, dim}})       where {order, dim}       = Any

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
Base.size(::Type{<:Tensor{3,dim}})          where {dim} = (dim, dim, dim)
Base.size(::Type{<:FourthOrderTensor{dim}}) where {dim} = (dim, dim, dim, dim)
Base.size(::Type{<:MixedTensor{1, Tuple{d1}}}) where {d1} = (d1,)
Base.size(::Type{<:MixedTensor2{d1, d2}}) where {d1, d2} = (d1, d2)
Base.size(::Type{<:MixedTensor3{d1, d2, d3}}) where {d1, d2, d3} = (d1, d2, d3)
Base.size(::Type{<:MixedTensor4{d1, d2, d3, d4}}) where {d1, d2, d3, d4} = (d1, d2, d3, d4)

# Also define length for the type itself
Base.length(::Type{Tensor{order, dim, T}}) where {order, dim, T} = n_components(Tensor{order, dim})

#########################
# Internal constructors #
#########################
for (TensorType, orders) in ((SymmetricTensor, (2,4)), (Tensor, (2,3,4)))
    for order in orders, dim in (1, 2, 3)
        N = n_components(TensorType{order, dim})
        @eval begin
            @inline $TensorType{$order, $dim}(t::NTuple{$N, T}) where {T} = $TensorType{$order, $dim, T}(t)
        end
        if N > 1 # To avoid overwriting ::Tuple{Any}
            # Heterogeneous tuple
            @eval @inline $TensorType{$order, $dim}(t::Tuple{Vararg{Any,$N}}) = $TensorType{$order, $dim}(promote(t...))
        end
    end
    if TensorType == Tensor
        for dim in (1, 2, 3)
            @eval @inline Tensor{1, $dim}(t::NTuple{$dim, T}) where {T} = Tensor{1, $dim, T}(t)
            if dim > 1 # To avoid overwriting ::Tuple{Any}
                # Heterogeneous tuple
                @eval @inline Tensor{1, $dim}(t::Tuple{Vararg{Any,$dim}}) = Tensor{1, $dim}(promote(t...))
            end
        end
    end
end
# Special for Vec (`Vec{dim}` is the same type as `Tensor{1, dim}`, so the
# generic `Tensor` constructors cover `Vec{dim}(data)`)
@inline Vec(data::NTuple{N}) where {N} = Vec{N}(data)
@inline Vec(data::Vararg{T,N}) where {T, N} = Vec{N,T}(data)

# General fallbacks (Tuples of the right length hit the default constructor, which
# converts to the computed NTuple field type)
@inline          Tensor{order, dim, T}(data::Union{AbstractArray, Function}) where {order, dim, T} = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline SymmetricTensor{order, dim, T}(data::Union{AbstractArray, Function}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, SymmetricTensor{order, dim}(data))

include("mixed_tensors.jl")
include("indexing.jl")
include("utilities.jl")
include("tensor_ops_errors.jl")
include("automatic_differentiation.jl")
include("promotion_conversion.jl")
include("constructors.jl")
include("basic_operations.jl")
include("tensor_products.jl")
include("transpose.jl")
include("symmetric.jl")
include("math_ops.jl")
include("eigen.jl")
include("special_ops.jl")
include("simd.jl")
include("voigt.jl")
include("precompile.jl")

end # module
