# Type constructors e.g. Tensor{2, 3}(arg)

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
        println("BAD DISPATCH")
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
        println("BAD DISPATCH")
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
        @inline Base.one{order, dim}(::Type{$(TensorType){order, dim}}) = one($TensorType{order, dim, Float64})
        @inline Base.one{order, dim, T, M}(::Type{$(TensorType){order, dim, T, M}}) = one($TensorType{order, dim, T})
        @inline Base.one{order, dim, T}(::$TensorType{order, dim, T}) = one($TensorType{order, dim, T})

        @generated function Base.one{order, dim, T}(S::Type{$(TensorType){order, dim, T}})
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
                z = zero(T)
                $ReturnTensor($exp)
            end
        end
    end
end

# zero, one, rand
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))))
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.$op{order, dim}(::Type{$TensorType{order, dim}}) = $op($TensorType{order, dim, Float64})
        @inline Base.$op{order, dim, T, N}(::Type{$TensorType{order, dim, T, N}}) = $op($TensorType{order, dim, T})
        @inline Base.$op{order, dim, T}(::Type{$TensorType{order, dim, T}}) = fill($el, $TensorType{order, dim})
    end
end
@eval @inline Base.$op{dim}(S::Type{Vec{dim}}) = $op(Vec{dim, Float64})
@eval @inline Base.$op(t::AllTensors) = $op(typeof(t))
end

@generated function Base.fill{T <: AbstractTensor}(el::Union{Number, Function}, S::Type{T})
    TensorType = get_base(get_type(S))
    N = n_components(TensorType)
    ele = el <: Number ? :(el) : :(el())
    expr = Expr(:tuple, [ele for i in 1:N]...)
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($expr)
    end
end

# Array with zero/ones
@inline Base.zeros{T <: AbstractTensor}(::Type{T}, dims...) = fill(zero(T), dims...)
@inline Base.ones{T <: AbstractTensor}(::Type{T}, dims...) = fill(one(T), dims...)

# diagm
@generated function Base.diagm{T <: SecondOrderTensor}(S::Type{T}, v::Union{AbstractVector, Tuple})
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
@inline Base.diagm{dim, T <: Number}(::Type{Tensor{2, dim}}, v::T) = v * one(Tensor{2, dim, T})
@inline Base.diagm{dim, T <: Number}(::Type{SymmetricTensor{2, dim}}, v::T) = v * one(SymmetricTensor{2, dim, T})

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
([1.0, 0.0], [0.0, 1.0])

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
