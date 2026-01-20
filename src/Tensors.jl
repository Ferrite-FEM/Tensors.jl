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

export AbstractTensor, SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

export otimes, ⊗, ⊡, dcontract, dev, vol, symmetric, skew, minorsymmetric, majorsymmetric
export otimesu, otimesl
export minortranspose, majortranspose, isminorsymmetric, ismajorsymmetric
export tdot, dott, dotdot
export hessian, gradient, curl, divergence, laplace
export @implement_gradient, propagate_gradient
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
    SymmetricTensor{order, dim, T, M}(data::NTuple) where {order, dim, T, M} = new{order, dim, T, M}(data)
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
struct Tensor{order, dim, T, M} <: AbstractTensor{order, dim, T}
    data::NTuple{M, T}
    Tensor{order, dim, T, M}(data::NTuple) where {order, dim, T, M} = new{order, dim, T, M}(data)
end

###############
# Typealiases #
###############
const Vec{dim, T, M} = Tensor{1, dim, T, dim}

const AllTensors{dim, T} = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                 SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                 Vec{dim, T}, Tensor{3, dim, T}}


const SecondOrderTensor{dim, T}   = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}}
const FourthOrderTensor{dim, T}   = Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}}
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

@pure get_type(::Type{Type{X}}) where {X} = X

@pure get_base(::Type{<:Tensor{order, dim}})          where {order, dim} = Tensor{order, dim}
@pure get_base(::Type{<:SymmetricTensor{order, dim}}) where {order, dim} = SymmetricTensor{order, dim}

@pure Base.eltype(::Type{Tensor{order, dim, T, M}})          where {order, dim, T, M} = T
@pure Base.eltype(::Type{Tensor{order, dim, T}})             where {order, dim, T}    = T
@pure Base.eltype(::Type{Tensor{order, dim}})                where {order, dim}       = Any
@pure Base.eltype(::Type{SymmetricTensor{order, dim, T, M}}) where {order, dim, T, M} = T
@pure Base.eltype(::Type{SymmetricTensor{order, dim, T}})    where {order, dim, T}    = T
@pure Base.eltype(::Type{SymmetricTensor{order, dim}})       where {order, dim}       = Any


############################
# Abstract Array interface #
############################
Base.IndexStyle(::Type{<:SymmetricTensor}) = IndexCartesian()
Base.IndexStyle(::Type{<:Tensor}) = IndexLinear()

########
# Size #
########
Base.size(::Vec{dim})               where {dim} = (dim,)
Base.size(::SecondOrderTensor{dim}) where {dim} = (dim, dim)
Base.size(::Tensor{3,dim})          where {dim} = (dim, dim, dim)
Base.size(::FourthOrderTensor{dim}) where {dim} = (dim, dim, dim, dim)

# Also define length for the type itself
Base.length(::Type{Tensor{order, dim, T, M}}) where {order, dim, T, M} = M

#########################
# Internal constructors #
#########################
for (TensorType, orders) in ((SymmetricTensor, (2,4)), (Tensor, (2,3,4)))
    for order in orders, dim in (1, 2, 3)
        N = n_components(TensorType{order, dim})
        @eval begin
            @inline $TensorType{$order, $dim}(t::NTuple{$N, T}) where {T} = $TensorType{$order, $dim, T, $N}(t)
            @inline $TensorType{$order, $dim, T1}(t::NTuple{$N, T2}) where {T1, T2} = $TensorType{$order, $dim, T1, $N}(t)
        end
        if N > 1 # To avoid overwriting ::Tuple{Any}
            # Heterogeneous tuple
            @eval @inline $TensorType{$order, $dim}(t::Tuple{Vararg{Any,$N}}) = $TensorType{$order, $dim}(promote(t...))
        end
    end
    if TensorType == Tensor
        for dim in (1, 2, 3)
            @eval @inline Tensor{1, $dim}(t::NTuple{$dim, T}) where {T} = Tensor{1, $dim, T, $dim}(t)
            if dim > 1 # To avoid overwriting ::Tuple{Any}
                # Heterogeneous tuple
                @eval @inline Tensor{1, $dim}(t::Tuple{Vararg{Any,$dim}}) = Tensor{1, $dim}(promote(t...))
            end
        end
    end
end
# Special for Vec
@inline Vec{dim}(data) where {dim} = Tensor{1, dim}(data)
@inline Vec(data::NTuple{N}) where {N} = Vec{N}(data)
@inline Vec(data::Vararg{T,N}) where {T, N} = Vec{N,T}(data)

# General fallbacks
@inline          Tensor{order, dim, T}(data::Union{AbstractArray, Tuple, Function}) where {order, dim, T} = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline SymmetricTensor{order, dim, T}(data::Union{AbstractArray, Tuple, Function}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, SymmetricTensor{order, dim}(data))
@inline          Tensor{order, dim, T, M}(data::Union{AbstractArray, Tuple, Function})  where {order, dim, T, M} = Tensor{order, dim, T}(data)
@inline SymmetricTensor{order, dim, T, M}(data::Union{AbstractArray, Tuple, Function})  where {order, dim, T, M} = SymmetricTensor{order, dim, T}(data)

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
