__precompile__()

module Tensors

import Base.@pure
using StaticArrays

export AbstractTensor, SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

export otimes, ⊗, ⊡, dcontract, dev, vol, symmetric, skew, minorsymmetric, majorsymmetric
export minortranspose, majortranspose, isminorsymmetric, ismajorsymmetric
export tdot, dott, dotdot
export hessian#, gradient
export basevec, eᵢ
export rotate


@deprecate extract_components(tensor) Array(tensor)

#########
# Types #
#########
abstract AbstractTensor{order, dim, T <: Real} <: AbstractArray{T, order}

immutable SymmetricTensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::SVector{M, T}
end

immutable Tensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::SVector{M, T}
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
get_data(t::AbstractTensor) = t.data.data
tovector(t::AbstractTensor) = t.data

function tomatrix{dim}(t::Tensor{4, dim})
    N = n_components(Tensor{2,dim})
    return SMatrix{N, N}(tovector(t))
end

function tomatrix{dim}(t::Tensor{2, dim})
    N = n_components(Tensor{1,dim})
    return SMatrix{N, N}(tovector(t))
end

@pure n_components{dim}(::Type{SymmetricTensor{2, dim}}) = dim*dim - div((dim-1)*dim, 2)
@pure function n_components{dim}(::Type{SymmetricTensor{4, dim}})
    n = n_components(SymmetricTensor{2, dim})
    return n*n
end
@pure n_components{order, dim}(::Type{Tensor{order, dim}}) = dim^order

@pure get_base{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor{order, dim}
@pure get_base{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor{order, dim}
@pure get_base{order, dim}(::Type{SymmetricTensor{order, dim}}) = SymmetricTensor{order, dim}
@pure get_base{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor{order, dim}
@pure get_base{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor{order, dim}
@pure get_base{order, dim}(::Type{Tensor{order, dim}}) = Tensor{order, dim}

############################
# Abstract Array interface #
############################
Base.linearindexing{T <: SymmetricTensor}(::Type{T}) = Base.LinearSlow()
Base.linearindexing{T <: Tensor}(::Type{T}) = Base.LinearFast()

get_type{X}(::Type{Type{X}}) = X

########
# Size #
########
Base.size{dim}(::Vec{dim}) = (dim,)
Base.size{dim}(::SecondOrderTensor{dim}) = (dim, dim)
Base.size{dim}(::FourthOrderTensor{dim}) = (dim, dim, dim, dim)

#########################
# Internal constructors #
#########################
function _constructor_check{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}})
    !(order in (1,2,4)) && throw(ArgumentError("tensors only supported for order 1, 2 and 4"))
    !(dim in (1,2,3)) && throw(ArgumentError("tensors only supported for dim 1, 2 and 3"))
    order == 1 && Tt <: SymmetricTensor && throw(ArgumentError("symmetric tensors only supported for order 2 and 4"))
    return nothing
end

for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline (::Type{$TensorType{order, dim}}){order, dim, T <: Real, M}(t::NTuple{M, T}) = $TensorType{order, dim}(SVector{M,T}(t))
        @inline (::Type{$TensorType{order, dim, T1}}){order, dim, T1 <: Real, T2 <: Real, M}(t::NTuple{M, T2}) = $TensorType{order, dim}(SVector{M,T1}(t))
        @inline (::Type{$TensorType{order, dim, T1, M}}){order, dim, T1 <: Real, T2 <: Real, M}(t::NTuple{M, T2}) = $TensorType{order, dim}(SVector{M,T1}(t))

        @inline (::Type{$TensorType{order, dim}}){order, dim, T <: Real, M}(t::SVector{M, T}) = _constructor_check($TensorType{order, dim})
    end
    for order in (2,4), dim in (1,2,3)
        M = n_components(TensorType{order,dim})
        @eval @inline (::Type{$TensorType{$order, $dim}}){T <: Real}(t::SVector{$M, T}) = $TensorType{$order, $dim, T, $M}(t)
        @eval @inline (::Type{$TensorType{$order, $dim}}){T <: Real, Q}(t::SMatrix{Q, Q, T, $M}) = $TensorType{$order, $dim, T, $M}(t.data)
    end
    if TensorType == Tensor
        for dim in (1,2,3)
            @eval @inline (::Type{Tensor{1, $dim}}){T <: Real}(t::SVector{$dim, T}) = Tensor{1, $dim, T, $dim}(t)
        end
    end
end
# Special for Vec
@inline (Tt::Type{Vec{dim}}){dim}(data) = Tensor{1, dim}(data)

# General fallbacks
@inline          (Tt::Type{Tensor{order, dim, T}}){order, dim, T}(data::Union{AbstractArray, Tuple, Function}) = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline (Tt::Type{SymmetricTensor{order, dim, T}}){order, dim, T}(data::Union{AbstractArray, Tuple, Function}) = convert(SymmetricTensor{order, dim, T}, SymmetricTensor{order, dim}(data))
#@inline          (Tt::Type{Tensor{order, dim, T, M}}){order, dim, T, M}(data::Union{AbstractArray, Tuple, Function}) = Tensor{order, dim, T}(data)
#@inline (Tt::Type{SymmetricTensor{order, dim, T, M}}){order, dim, T, M}(data::Union{AbstractArray, Tuple, Function}) = SymmetricTensor{order, dim, T}(data)

###############
# Simple Math #
###############

# op(Number, Tensor): *, /
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.:*{order, dim, T, N}(n::Number, t::$TensorType{order, dim, T, N}) = $TensorType{order, dim}(n * tovector(t))
        @inline Base.:*{order, dim, T, N}(t::$TensorType{order, dim, T, N}, n::Number) = $TensorType{order, dim}(tovector(t) * n)
        @inline Base.:/{order, dim, T, N}(t::$TensorType{order, dim, T, N}, n::Number) = $TensorType{order, dim}(tovector(t) / n)

        # Unary -, +
        @inline Base.:-{order, dim, T, N}(t::$TensorType{order, dim, T, N}) = $TensorType{order, dim}(-tovector(t))
        @inline Base.:+{order, dim, T, N}(t::$TensorType{order, dim, T, N}) = $TensorType{order, dim}(+tovector(t))
    end
end

# Binary op(Tensor, Tensor): +, -, .+, .-, .*, ./
for op in (:+, :-, :.+, :.-, :.*, :./)
    for TensorType in (SymmetricTensor, Tensor)
        @eval begin
            @inline function Base.$op{order, dim, T1, T2, N}(t1::$TensorType{order, dim, T1, N}, t2::$TensorType{order, dim, T2, N})
                $TensorType{order, dim}($op(tovector(t1), tovector(t2)))
            end
        end
    end
    @eval begin
        Base.$op{order, dim}(t1::AbstractTensor{order, dim}, t2::AbstractTensor{order, dim}) = $op(promote(t1, t2)...)
    end
end

######################
# Basic constructors #
######################

# zero, rand, ones
for op in (:zero, :rand, :ones)
    for TensorType in (SymmetricTensor, Tensor)
        @eval begin
            @inline Base.$op{order, dim}(Tt::Type{$TensorType{order, dim}}) = $op($TensorType{order, dim, Float64})
            @inline Base.$op{order, dim, T, M}(Tt::Type{$TensorType{order, dim, T, M}}) = $op($TensorType{order, dim, T})
            @inline function Base.$op{order, dim, T}(Tt::Type{$TensorType{order, dim, T}})
                N = n_components($TensorType{order, dim})
                return $TensorType{order, dim}($op(SVector{N, T}))
            end
        end
    end
    # Special case for Vec
    @eval @inline Base.$op{dim}(Tt::Type{Vec{dim}}) = $op(Vec{dim, Float64})

    # zero, rand or ones of a tensor
    @eval @inline Base.$op(t::AllTensors) = $op(typeof(t))
end

# zeros, ones
for (op, el) in ((:zeros, :zero), (:ones, :one))
    for TensorType in (SymmetricTensor, Tensor)
        @eval begin
            @inline Base.$op{order, dim}(Tt::Type{$TensorType{order, dim}}, dims...) = $op($TensorType{order, dim, Float64}, dims...)
            @inline function Base.$op{order, dim, T}(Tt::Type{$TensorType{order, dim, T}}, dims...)
                N = n_components($TensorType{order, dim})
                return $op($TensorType{order, dim, T, N}, dims...)
            end
            @inline Base.$op{order, dim, T, M}(Tt::Type{$TensorType{order, dim, T, M}}, dims...) =
                fill!(Array{$TensorType{order, dim, T, M}}(dims...), $el($TensorType{order, dim, T}))
        end
    end
    @eval @inline Base.$op{dim}(::Type{Vec{dim}}, dims...) = $op(Vec{dim, Float64}, dims...)
end

# diagm
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @generated function Base.diagm{dim}(Tt::Type{$(TensorType){2, dim}}, v::Union{AbstractVector, Tuple})
            f = (i,j) -> i == j ? :(v[$i]) : :($(zero(eltype(v))))
            exp = tensor_create(get_type(Tt), f)
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $exp
                $($TensorType){2, dim}(t)
            end
        end

        @inline Base.diagm{dim, T}(Tt::Type{$(TensorType){2, dim}}, v::T) = v * one($(TensorType){2, dim, T})
    end
end

# one (identity tensor)
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.one{order, dim}(Tt::Type{$(TensorType){order, dim}}) = one($TensorType{order, dim, Float64})
        @inline Base.one{order, dim, T, M}(Tt::Type{$(TensorType){order, dim, T, M}}) = one($TensorType{order, dim, T})
        @inline Base.one{order, dim, T}(Tt::$TensorType{order, dim, T}) = one($TensorType{order, dim, T})

        @generated function Base.one{order, dim, T}(Tt::Type{$(TensorType){order, dim, T}})
            !(order in (2,4)) && throw(ArgumentError("`one` only defined for order 2 and 4"))
            δ = (i,j) -> i == j ? :($(one(T))) : :($(zero(T)))
            if order == 2
                f = (i,j) -> δ(i,j)
            elseif order == 4 && $TensorType == Tensor
                f = (i,j,k,l) -> δ(i,k) * δ(j,l)
            else # order == 4 && TensorType == SymmetricTensor
                f = (i,j,k,l) -> (δ(i,k) * δ(j,l) + δ(i,l) * δ(j,k)) / 2
            end
            exp = tensor_create(get_base(get_type(Tt)), f)
            return quote
                $(Expr(:meta, :inline))
                $($TensorType){order, dim}($exp)
            end
        end
    end
end

# Tensor from function
@generated function (Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}){order, dim}(f::Function)
    _constructor_check(get_base(get_type(Tt)))
    if order == 1
        exp = tensor_create(get_base(get_type(Tt)), (i) -> :(f($i)))
    elseif order == 2
        exp = tensor_create(get_base(get_type(Tt)), (i,j) -> :(f($i, $j)))
    elseif order == 4
        exp = tensor_create(get_base(get_type(Tt)), (i,j,k,l) -> :(f($i, $j, $k, $l)))
    end

    return :(get_base(Tt)($exp))
end
Base.convert{dim, T}(::Type{SVector{dim, T}}, f::Function) = SVector{dim, T}(ntuple(f, Val{dim}))

# Tensor from AbstractArray
@generated function (Tt::Type{Tensor{order, dim}}){order, dim}(data::AbstractArray)
    _constructor_check(get_base(get_type(Tt)))
    n = n_components(Tensor{order,dim})
    return quote
        if length(data) != $n
            throw(ArgumentError("wrong number of vector elements, expected $($n), got $(length(data))"))
        end
        Tensor{order, dim}(SVector{$n}(data))
    end
end

# SymmetricTensor from AbstractArray
@generated function (Tt::Type{SymmetricTensor{order, dim}}){order, dim}(data::AbstractArray)
    _constructor_check(get_base(get_type(Tt)))
    n = n_components(Tensor{order,dim})
    m = n_components(SymmetricTensor{order,dim})
    return quote
        if length(data) != $n && length(data) != $m
            throw(ArgumentError("wrong number of vector elements, expected $($n) or $($m), got $(length(data))"))
        end
        if length(data) == $m
            return SymmetricTensor{order, dim}(SVector{$m}(data))
        end
        S = Tensor{order, dim}(SVector{$n}(data))
        return convert(SymmetricTensor{order, dim}, S)
    end
end

"""
```julia
basevec(::Type{Vec{dim, T}})
basevec(::Type{Vec{dim, T}}, i)
basevec(::Vec{dim, T})
basevec(::Vec{dim, T}, i)
```
Return a tuple with the base vectors corresponding to the dimension `dim` and type
`T`. An optional integer `i` can be used to extract the i:th base vector.
The alias `eᵢ` can also be used, written `e\\_i<TAB>`.

**Example:**

```jldoctest
julia> eᵢ(Vec{2, Float64})
([1.0,0.0],[0.0,1.0])

julia> eᵢ(Vec{2, Float64}, 2)
2-element Tensors.Tensor{1,2,Float64,2}:
 0.0
 1.0
```
"""
@inline function basevec{T}(::Type{Vec{1, T}})
    o = one(T)
    return (Vec{1, T}((o,)), )
end
@inline function basevec{T}(::Type{Vec{2, T}})
    o = one(T)
    z = zero(T)
    return (Vec{2, T}((o, z)),
            Vec{2, T}((z, o)))
end
@inline function basevec{T}(::Type{Vec{3, T}})
    o = one(T)
    z = zero(T)
    return (Vec{3, T}((o, z, z)),
            Vec{3, T}((z, o, z)),
            Vec{3, T}((z, z, o)))
end

@inline basevec{dim}(::Type{Vec{dim}}) = basevec(Vec{dim, Float64})
@inline basevec{dim, T}(::Type{Vec{dim, T}}, i::Int) = basevec(Vec{dim, T})[i]
@inline basevec{dim}(::Type{Vec{dim}}, i::Int) = basevec(Vec{dim, Float64})[i]
@inline basevec{dim, T}(v::Vec{dim, T}) = basevec(typeof(v))
@inline basevec{dim, T}(v::Vec{dim, T}, i::Int) = basevec(typeof(v), i)

const eᵢ = basevec

end # module
