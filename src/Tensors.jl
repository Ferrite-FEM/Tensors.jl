__precompile__()

module Tensors

import Base.@pure
using Compat

export AbstractTensor, SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

export otimes, ⊗, ⊡, dcontract, dev, vol, symmetric, skew, minorsymmetric, majorsymmetric
export minortranspose, majortranspose, isminorsymmetric, ismajorsymmetric
export tdot, dott, dotdot
export hessian#, gradient
export basevec, eᵢ
export rotate

#########
# Types #
#########
@compat abstract type AbstractTensor{order, dim, T <: Real} <: AbstractArray{T, order} end

immutable SymmetricTensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

immutable Tensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

###############
# Typealiases #
###############
@compat const Vec{dim, T, M} = Tensor{1, dim, T, dim}

@compat const AllTensors{dim, T} = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                         SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                         Vec{dim, T}}


@compat const SecondOrderTensor{dim, T}   = Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}}
@compat const FourthOrderTensor{dim, T}   = Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}}
@compat const SymmetricTensors{dim, T}    = Union{SymmetricTensor{2, dim, T}, SymmetricTensor{4, dim, T}}
@compat const NonSymmetricTensors{dim, T} = Union{Tensor{2, dim, T}, Tensor{4, dim, T}, Vec{dim, T}}


##############################
# Utility/Accessor Functions #
##############################
get_data(t::AbstractTensor) = t.data

@pure n_components{dim}(::Type{SymmetricTensor{2, dim}}) = dim*dim - div((dim-1)*dim, 2)
@pure function n_components{dim}(::Type{SymmetricTensor{4, dim}})
    n = n_components(SymmetricTensor{2, dim})
    return n*n
end
@pure n_components{order, dim}(::Type{Tensor{order, dim}}) = dim^order

@pure get_type{X}(::Type{Type{X}}) = X

@pure get_base{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor{order, dim}
@pure get_base{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor{order, dim}
@pure get_base{order, dim}(::Type{Tensor{order, dim}}) = Tensor{order, dim}
@pure get_base{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor{order, dim}
@pure get_base{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor{order, dim}
@pure get_base{order, dim}(::Type{SymmetricTensor{order, dim}}) = SymmetricTensor{order, dim}

@pure Base.eltype{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = T
@pure Base.eltype{order, dim, T}(::Type{Tensor{order, dim, T}}) = T
@pure Base.eltype{order, dim}(::Type{Tensor{order, dim}}) = Any
@pure Base.eltype{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = T
@pure Base.eltype{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = T
@pure Base.eltype{order, dim}(::Type{SymmetricTensor{order, dim}}) = Any


############################
# Abstract Array interface #
############################
@compat Base.IndexStyle(::Type{<: SymmetricTensor}) = IndexCartesian()
@compat Base.IndexStyle(::Type{<: Tensor}) = IndexLinear()

########
# Size #
########
Base.size{dim}(::Vec{dim}) = (dim,)
Base.size{dim}(::SecondOrderTensor{dim}) = (dim, dim)
Base.size{dim}(::FourthOrderTensor{dim}) = (dim, dim, dim, dim)

#########################
# Internal constructors #
#########################
for TensorType in (SymmetricTensor, Tensor)
    for order in (2, 4), dim in (1, 2, 3)
        N = n_components(TensorType{order, dim})
        @eval begin
            @inline (::Type{$TensorType{$order, $dim}}){T <: Real}(t::NTuple{$N, T}) = $TensorType{$order, $dim, T, $N}(t)
            @inline (::Type{$TensorType{$order, $dim, T1}}){T1 <: Real, T2 <: Real}(t::NTuple{$N, T2}) = $TensorType{$order, $dim, T1, $N}(t)
        end
    end
    if TensorType == Tensor
        for dim in (1, 2, 3)
            @eval @inline (::Type{Tensor{1, $dim}}){T <: Real}(t::NTuple{$dim, T}) = Tensor{1, $dim, T, $dim}(t)
        end
    end
end
# Special for Vec
@inline (Tt::Type{Vec{dim}}){dim}(data) = Tensor{1, dim}(data)
Base.convert{dim, T}(::Type{NTuple{dim, T}}, f::Function) = NTuple{dim, T}(ntuple(f, Val{dim}))


# General fallbacks
@inline          (Tt::Type{Tensor{order, dim, T}}){order, dim, T}(data::Union{AbstractArray, Tuple, Function}) = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline (Tt::Type{SymmetricTensor{order, dim, T}}){order, dim, T}(data::Union{AbstractArray, Tuple, Function}) = convert(SymmetricTensor{order, dim, T}, SymmetricTensor{order, dim}(data))
#@inline          (Tt::Type{Tensor{order, dim, T, M}}){order, dim, T, M}(data::Union{AbstractArray, Tuple, Function}) = Tensor{order, dim, T}(data)
#@inline (Tt::Type{SymmetricTensor{order, dim, T, M}}){order, dim, T, M}(data::Union{AbstractArray, Tuple, Function}) = SymmetricTensor{order, dim, T}(data)

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
include("special_ops.jl")

end # module
