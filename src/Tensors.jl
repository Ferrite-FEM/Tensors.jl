__precompile__()

module Tensors

import Base.@pure

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
abstract AbstractTensor{order, dim, T <: Real} <: AbstractArray{T, order}

immutable SymmetricTensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

immutable Tensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

###############
# Typealiases #
###############
typealias Vec{dim, T, M} Tensor{1, dim, T, dim}

typealias AllTensors{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                   SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                   Vec{dim, T}}


typealias SecondOrderTensor{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}}
typealias FourthOrderTensor{dim, T} Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}}
typealias SymmetricTensors{dim, T} Union{SymmetricTensor{2, dim, T}, SymmetricTensor{4, dim, T}}
typealias NonSymmetricTensors{dim, T} Union{Tensor{2, dim, T}, Tensor{4, dim, T}, Vec{dim, T}}

include("indexing.jl")
include("promotion_conversion.jl")
include("constructors.jl")
include("basic_operations.jl")
include("tensor_products.jl")
include("utilities.jl")
include("transpose.jl")
include("symmetric.jl")
include("math_ops.jl")
include("special_ops.jl")
include("tensor_ops_errors.jl")
include("automatic_differentiation.jl")

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
Base.linearindexing{T <: SymmetricTensor}(::Type{T}) = Base.LinearSlow()
Base.linearindexing{T <: Tensor}(::Type{T}) = Base.LinearFast()

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

############################################
# Type constructors e.g. Tensor{2, 3}(arg) #
############################################

# Tensor from function
@generated function (S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}){order, dim}(f::Function)
    TensorType = get_base(get_type(S))
    if order == 1
        exp = tensor_create(TensorType, (i) -> :(f($i)))
    elseif order == 2
        exp = tensor_create(TensorType, (i,j) -> :(f($i, $j)))
    elseif order == 4
        exp = tensor_create(TensorType, (i,j,k,l) -> :(f($i, $j, $k, $l)))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

# Tensor from AbstractArray
@generated function (::Type{Tensor{order, dim}}){order, dim}(data::AbstractArray)
    N = n_components(Tensor{order,dim})
    exp = Expr(:tuple, [:(data[$i]) for i in 1:N]...)
    return quote
        if length(data) != $N
            throw(ArgumentError("wrong number of elements, expected $($N), got $(length(data))"))
        end
        Tensor{order, dim}($exp)
    end
end

# SymmetricTensor from AbstractArray
@generated function (::Type{SymmetricTensor{order, dim}}){order, dim}(data::AbstractArray)
    N = n_components(Tensor{order,dim})
    expN = Expr(:tuple, [:(data[$i]) for i in 1:N]...)
    M = n_components(SymmetricTensor{order,dim})
    expM = Expr(:tuple, [:(data[$i]) for i in 1:M]...)
    return quote
        L = length(data)
        if L != $N && L != $M
            throw(ArgumentError("wrong number of vector elements, expected $($N) or $($M), got $L"))
        end
        if L == $M
            @inbounds return SymmetricTensor{order, dim}($expM)
        end
        @inbounds S = Tensor{order, dim}($expN)
        return convert(SymmetricTensor{order, dim}, S)
    end
end

end # module
