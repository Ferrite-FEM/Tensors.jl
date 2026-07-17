# Type constructors e.g. Tensor{2, 3}(arg)

######################
# Tuple constructors #
######################
# Definition-time generated methods for every supported shape, so that a
# wrong-length tuple gives a natural MethodError.
for (TensorType, orders) in ((SymmetricTensor, (2, 4)), (Tensor, (2, 3, 4)))
    for order in orders, dim in (1, 2, 3)
        N = n_components(TensorType{order, dim})
        @eval begin
            @inline $TensorType{$order, $dim}(t::NTuple{$N, T}) where {T} = $TensorType{$order, $dim, T, $N}(t)
            @inline $TensorType{$order, $dim, T1}(t::NTuple{$N, T2}) where {T1, T2} = $TensorType{$order, $dim, T1, $N}(t)
        end
        if N > 1 # To avoid overwriting ::Tuple{Any}
            # Heterogeneous tuple
            @eval @inline $TensorType{$order, $dim}(t::Tuple{Vararg{Any, $N}}) = $TensorType{$order, $dim}(promote(t...))
        end
    end
    if TensorType == Tensor
        for dim in (1, 2, 3)
            @eval @inline Tensor{1, $dim}(t::NTuple{$dim, T}) where {T} = Tensor{1, $dim, T, $dim}(t)
            if dim > 1 # To avoid overwriting ::Tuple{Any}
                # Heterogeneous tuple
                @eval @inline Tensor{1, $dim}(t::Tuple{Vararg{Any, $dim}}) = Tensor{1, $dim}(promote(t...))
            end
        end
    end
end

# MixedTensor (`dims <: Tuple` and the structural invariants — one dimension
# per index, component count matching — are enforced here, see the struct
# definition for why they cannot live on the struct itself)
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
@inline MixedTensor{order, dims}(data::Tuple{Vararg{Any, M}}) where {order, dims <: Tuple, M} = MixedTensor{order, dims}(promote(data...))

# Special for Vec
@inline Vec{dim}(data) where {dim} = Tensor{1, dim}(data)
@inline Vec(data::NTuple{N}) where {N} = Vec{N}(data)
@inline Vec(data::Vararg{T, N}) where {T, N} = Vec{N, T}(data)

# General fallbacks
@inline          Tensor{order, dim, T}(data::Union{AbstractArray, Tuple, Function}) where {order, dim, T} = convert(Tensor{order, dim, T}, Tensor{order, dim}(data))
@inline SymmetricTensor{order, dim, T}(data::Union{AbstractArray, Tuple, Function}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, SymmetricTensor{order, dim}(data))
# NOTE: `Tuple` is deliberately not in these fallbacks: tuple data hits the
# default constructor (which converts elementwise to NTuple{M, T}), and
# including it here would shadow that constructor and recurse.
@inline          Tensor{order, dim, T, M}(data::Union{AbstractArray, Function})  where {order, dim, T, M} = Tensor{order, dim, T}(data)
@inline SymmetricTensor{order, dim, T, M}(data::Union{AbstractArray, Function})  where {order, dim, T, M} = SymmetricTensor{order, dim, T}(data)

########################
# Tensor from function #
########################
@generated function (S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}, Type{MixedTensor{order, dim}}})(f::Function) where {order, dim}
    TensorType = get_base(get_type(S))
    exp = component_expr(TensorType, (inds...) -> :(f($(inds...))))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

#############################
# Tensor from AbstractArray #
#############################
function Tensor{order, dim}(data::AbstractArray) where {order, dim}
    N = n_components(Tensor{order, dim})
    length(data) != N && throw(ArgumentError("wrong number of elements, expected $N, got $(length(data))"))
    return apply_all(Tensor{order, dim}, @inline function(i) @inbounds data[i]; end)
end

@generated function SymmetricTensor{order, dim}(data::AbstractArray) where {order, dim}
    N = n_components(Tensor{order, dim})
    expN = Expr(:tuple, [:(data[$i]) for i in 1:N]...)
    M = n_components(SymmetricTensor{order, dim})
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

#########################
# one (identity tensor) #
#########################
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.one(::Type{$(TensorType){order, dim}}) where {order, dim} = one($TensorType{order, dim, Float64})
        @inline Base.one(::Type{$(TensorType){order, dim, T, M}}) where {order, dim, T, M} = one($TensorType{order, dim, T})
        @inline Base.one(::$TensorType{order, dim, T}) where {order, dim, T} = one($TensorType{order, dim, T})

        @generated function Base.one(S::Type{$(TensorType){order, dim, T}}) where {order, dim, T}
            !(order in (2, 4)) && return :(throw(ArgumentError("`one` only defined for order 2 and 4")))
            δ = (i, j) -> i == j ? :(o) : :(z)
            ReturnTensor = get_base(get_type(S))
            if order == 2
                f = (i, j) -> δ(i, j)
            elseif order == 4 && $TensorType == Tensor
                f = (i, j, k, l) -> :($(δ(i, k)) * $(δ(j, l)))
            else # order == 4 && TensorType == SymmetricTensor
                f = (i, j, k, l) -> :(($(δ(i, k)) * $(δ(j, l)) + $(δ(i, l)) * $(δ(j, k))) / 2)
            end
            exp = component_expr(ReturnTensor, f)
            return quote
                $(Expr(:meta, :inline))
                o = one(T)
                z = zero(o) # zero-no-unit(T)
                $ReturnTensor($exp)
            end
        end
    end
end

##########################
# zero, ones, rand, fill #
##########################
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(() -> rand(T))), (:randn, :(() -> randn(T))))
    for TensorType in (SymmetricTensor, Tensor, MixedTensor)
        @eval begin
            @inline Base.$op(::Type{$TensorType{order, dim}}) where {order, dim} = $op($TensorType{order, dim, Float64})
            @inline Base.$op(::Type{$TensorType{order, dim, T, N}}) where {order, dim, T, N} = $op($TensorType{order, dim, T})
            @inline Base.$op(::Type{$TensorType{order, dim, T}}) where {order, dim, T} = fill($el, $TensorType{order, dim})
        end
    end
    @eval @inline Base.$op(S::Type{Vec{dim}}) where {dim} = $op(Vec{dim, Float64})
    @eval @inline Base.$op(t::AbstractTensor) = $op(typeof(t))
end

@inline Base.fill(el::Number, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = apply_all(get_base(T), i -> el)
@inline Base.fill(f::Function, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = apply_all(get_base(T), i -> f())

# Array with zero/ones
@inline Base.zeros(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = fill(zero(T), (dims))
@inline Base.ones(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = fill(one(T), (dims))

#########
# diagm #
#########
@generated function LinearAlgebra.diagm(S::Type{T}, v::Union{AbstractVector, Tuple}) where {T <: SecondOrderTensor}
    TensorType = get_base(get_type(S))
    ET = eltype(get_type(S)) == Any ? eltype(v) : eltype(get_type(S))
    f = (i, j) -> i == j ? :($ET(v[$i])) : :(o)
    exp = component_expr(TensorType, f)
    return quote
        $(Expr(:meta, :inline))
        o = zero($ET)
        @inbounds return $TensorType($exp)
    end
end
@inline LinearAlgebra.diagm(::Type{Tensor{2, dim}}, v::T) where {dim, T <: Number} = v * one(Tensor{2, dim, T})
@inline LinearAlgebra.diagm(::Type{SymmetricTensor{2, dim}}, v::T) where {dim, T <: Number} = v * one(SymmetricTensor{2, dim, T})

"""
    basevec(::Type{Vec{dim, T}})
    basevec(::Type{Vec{dim, T}}, i)
    basevec(::Vec{dim, T})
    basevec(::Vec{dim, T}, i)

Return a tuple with the base vectors corresponding to the dimension `dim` and type
`T`. An optional integer `i` can be used to extract the i:th base vector.
The alias `eᵢ` can also be used, written `e\\_i<TAB>`.

# Examples
```jldoctest
julia> eᵢ(Vec{2, Float64})
([1.0, 0.0], [0.0, 1.0])

julia> eᵢ(Vec{2, Float64}, 2)
2-element Vec{2, Float64}:
 0.0
 1.0
```
"""
@generated function basevec(::Type{Vec{dim, T}}) where {dim, T}
    vecs = [Expr(:tuple, [i == j ? :o : :z for i in 1:dim]...) for j in 1:dim]
    return quote
        $(Expr(:meta, :inline))
        o = one(T)
        z = zero(T)
        return ($([:(Vec{$dim, T}($e)) for e in vecs]...),)
    end
end

@inline basevec(::Type{Vec{dim}}) where {dim} = basevec(Vec{dim, Float64})
@inline basevec(::Type{Vec{dim, T}}, i::Int) where {dim, T} = basevec(Vec{dim, T})[i]
@inline basevec(::Type{Vec{dim}}, i::Int) where {dim} = basevec(Vec{dim, Float64})[i]
@inline basevec(v::Vec{dim, T}) where {dim, T} = basevec(typeof(v))
@inline basevec(v::Vec{dim, T}, i::Int) where {dim, T} = basevec(typeof(v), i)

const eᵢ = basevec
