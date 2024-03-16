# Type constructors e.g. Tensor{2, 3}(arg)

# Tensor from function
@generated function (S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}})(f::Function) where {order, dim}
    TensorType = get_base(get_type(S))
    if order == 1
        exp = tensor_create(TensorType, (i) -> :(f($i)))
    elseif order == 2
        exp = tensor_create(TensorType, (i,j) -> :(f($i, $j)))
    elseif order == 3
        exp = tensor_create(TensorType, (i,j,k) -> :(f($i, $j, $k)))
    elseif order == 4
        exp = tensor_create(TensorType, (i,j,k,l) -> :(f($i, $j, $k, $l)))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

# Applies the function f to all indices f(1), f(2), ... f(n_independent_components)
@generated function apply_all(S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}, f::Function) where {order, dim}
    TensorType = get_base(get_type(S))
    exp = tensor_create_linear(TensorType, (i) -> :(f($i)))
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

@inline function apply_all(S::Union{Tensor{order, dim}, SymmetricTensor{order, dim}}, f::Function) where {order, dim}
    apply_all(get_base(typeof(S)), f)
end

# Tensor from AbstractArray
function Tensor{order, dim}(data::AbstractArray) where {order, dim}
    N = n_components(Tensor{order, dim})
    length(data) != n_components(Tensor{order, dim}) && throw(ArgumentError("wrong number of elements, expected $N, got $(length(data))"))
    return apply_all(Tensor{order, dim}, @inline function(i) @inbounds data[i]; end)
end


# SymmetricTensor from AbstractArray
@generated function SymmetricTensor{order, dim}(data::AbstractArray) where {order, dim}
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

# one (identity tensor)
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.one(::Type{$(TensorType){order, dim}}) where {order, dim} = one($TensorType{order, dim, Float64})
        @inline Base.one(::Type{$(TensorType){order, dim, T, M}}) where {order, dim, T, M} = one($TensorType{order, dim, T})
        @inline Base.one(::$TensorType{order, dim, T}) where {order, dim, T} = one($TensorType{order, dim, T})

        @generated function Base.one(S::Type{$(TensorType){order, dim, T}}) where {order, dim, T}
            !(order in (2,4)) && throw(ArgumentError("`one` only defined for order 2 and 4"))
            δ = (i,j) -> i == j ? :(o) : :(z)
            ReturnTensor = get_base(get_type(S))
            if order == 2
                f = (i,j) -> :($(δ(i,j)))
            elseif order == 4 && $TensorType == Tensor
                f = (i,j,k,l) -> :($(δ(i,k)) * $(δ(j,l)))
            else # order == 4 && TensorType == SymmetricTensor
                f = (i,j,k,l) -> :(($(δ(i,k)) * $(δ(j,l)) + $(δ(i,l))* $(δ(j,k))) / 2)
            end
            exp = tensor_create(ReturnTensor, f)
            return quote
                $(Expr(:meta, :inline))
                o = one(T)
                z = zero(o) # zero-no-unit(T)
                $ReturnTensor($exp)
            end
        end
    end
end

# zero, one, randn. 
# rand included here to make rand(::Type{AbstractTensor}) fast on julia 1.6. 
# When 1.6 support is dropped, the general implementation below can be used instead. 
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:randn,:(()->randn(T))), (:rand,:(()->rand(T))))
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.$op(::Type{$TensorType{order, dim}}) where {order, dim} = $op($TensorType{order, dim, Float64})
        @inline Base.$op(::Type{$TensorType{order, dim, T, N}}) where {order, dim, T, N} = $op($TensorType{order, dim, T})
        @inline Base.$op(::Type{$TensorType{order, dim, T}}) where {order, dim, T} = fill($el, $TensorType{order, dim})
    end
end
@eval @inline Base.$op(S::Type{Vec{dim}}) where {dim} = $op(Vec{dim, Float64})
@eval @inline Base.$op(t::AllTensors) = $op(typeof(t))
end

# Helper to construct a fully specified tensor from at least the specification returned by get_base
function default_concrete_tensor_type(::Type{X}) where {X <: Union{Tensor, SymmetricTensor}}
    TB = get_base(X)
    T = eltype(X) === Any ? Float64 : eltype(X)
    M = n_components(TB)
    return TB{T, M}
end

# For `rand`, hook into Random
function Random.rand(rng::Random.AbstractRNG, ::Random.SamplerType{TT}) where {TT <: Union{Tensor{order, dim}, SymmetricTensor{order, dim}}} where {order, dim}
    TC = default_concrete_tensor_type(TT)
    return apply_all(get_base(TT), _ -> rand(rng, eltype(TC)))::TC # typeassert needed on julia 1.6, but ok on 1.8 and later. 
end
# Always use the `SamplerType` as the value has no influence on the random generation.
Random.Sampler(::Type{<:Random.AbstractRNG}, t::AllTensors, ::Random.Repetition) = Random.SamplerType{typeof(t)}()
# Fix to make `rand([rng], ::Type{AbstractTensor}, d, dims...)` have a concrete eltype
function Random.rand(r::Random.AbstractRNG, ::Type{X}, dims::Dims) where {X <: Union{Tensor, SymmetricTensor}}
    TC = default_concrete_tensor_type(X)
    return Random.rand!(r, Array{TC}(undef, dims), X)
end

@inline Base.fill(el::Number, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor}} = apply_all(get_base(T), i -> el)
@inline Base.fill(f::Function, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor}} = apply_all(get_base(T), i -> f())

# Array with zero/ones
@inline Base.zeros(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor}} = fill(zero(T), (dims))
@inline Base.ones(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor}} = fill(one(T), (dims))

# diagm
@generated function LinearAlgebra.diagm(S::Type{T}, v::Union{AbstractVector, Tuple}) where {T <: SecondOrderTensor}
    TensorType = get_base(get_type(S))
    ET = eltype(get_type(S)) == Any ? eltype(v) : eltype(get_type(S)) # lol
    f = (i,j) -> i == j ? :($ET(v[$i])) : :(o)
    exp = tensor_create(TensorType, f)
    return quote
        $(Expr(:meta, :inline))
        o = zero($ET)
        @inbounds return $TensorType($exp)
    end
end
@inline LinearAlgebra.diagm(::Type{Tensor{2, dim}}, v::T) where {dim, T<:Number} = v * one(Tensor{2, dim, T})
@inline LinearAlgebra.diagm(::Type{SymmetricTensor{2, dim}}, v::T) where {dim, T<:Number} = v * one(SymmetricTensor{2, dim, T})

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
@inline function basevec(::Type{Vec{1, T}}) where {T}
    o = one(T)
    return (Vec{1, T}((o,)), )
end
@inline function basevec(::Type{Vec{2, T}}) where {T}
    o = one(T)
    z = zero(T)
    return (Vec{2, T}((o, z)),
            Vec{2, T}((z, o)))
end
@inline function basevec(::Type{Vec{3, T}}) where {T}
    o = one(T)
    z = zero(T)
    return (Vec{3, T}((o, z, z)),
            Vec{3, T}((z, o, z)),
            Vec{3, T}((z, z, o)))
end

@inline basevec(::Type{Vec{dim}}) where {dim} = basevec(Vec{dim, Float64})
@inline basevec(::Type{Vec{dim, T}}, i::Int) where {dim, T} = basevec(Vec{dim, T})[i]
@inline basevec(::Type{Vec{dim}}, i::Int) where {dim} = basevec(Vec{dim, Float64})[i]
@inline basevec(v::Vec{dim, T}) where {dim, T} = basevec(typeof(v))
@inline basevec(v::Vec{dim, T}, i::Int) where {dim, T} = basevec(typeof(v), i)

const eᵢ = basevec
