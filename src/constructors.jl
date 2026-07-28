# Type constructors e.g. Tensor{2, 3}(arg)

# Tensor from function
@generated function (S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}, Type{MixedTensor{order, dim}}})(f::Function) where {order, dim}
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

# Applies the function f to all linear indices: TT((f(1), f(2), ..., f(M))).
# (Base's tuple `map` is not usable here: it leaves the unrolled path beyond
# 32 components, e.g. Tensor{4, 3} with 81.)
@inline function apply_all(::Type{TT}, f::F) where {TT <: Union{Tensor, SymmetricTensor, MixedTensor}, F <: Function}
    B = get_base(TT)
    return B(ntuple(f, Val(n_components(B))))
end
@inline apply_all(S::AbstractTensor, f::F) where {F <: Function} = apply_all(get_base(typeof(S)), f)

# Tensor from AbstractArray
function Tensor{order, dim}(data::AbstractArray) where {order, dim}
    N = n_components(Tensor{order, dim})
    length(data) != n_components(Tensor{order, dim}) && throw(ArgumentError("wrong number of elements, expected $N, got $(length(data))"))
    return apply_all(Tensor{order, dim}, @inline function(i) @inbounds data[i]; end)
end


# SymmetricTensor from AbstractArray
function SymmetricTensor{order, dim}(data::AbstractArray) where {order, dim}
    N = n_components(Tensor{order, dim})
    M = n_components(SymmetricTensor{order, dim})
    L = length(data)
    if L == M
        return apply_all(SymmetricTensor{order, dim}, @inline function(i) @inbounds data[i]; end)
    elseif L == N
        return convert(SymmetricTensor{order, dim}, Tensor{order, dim}(data))
    end
    throw(ArgumentError("wrong number of vector elements, expected $N or $M, got $L"))
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

# zero, one, rand
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))), (:randn,:(()->randn(T))))
for TensorType in (SymmetricTensor, Tensor, MixedTensor)
    @eval begin
        @inline Base.$op(::Type{$TensorType{order, dim}}) where {order, dim} = $op($TensorType{order, dim, Float64})
        @inline Base.$op(::Type{$TensorType{order, dim, T, N}}) where {order, dim, T, N} = $op($TensorType{order, dim, T})
        @inline Base.$op(::Type{$TensorType{order, dim, T}}) where {order, dim, T} = fill($el, $TensorType{order, dim})
    end
end
@eval @inline Base.$op(S::Type{Vec{dim}}) where {dim} = $op(Vec{dim, Float64})
@eval @inline Base.$op(t::AllTensors) = $op(typeof(t))
end

@inline Base.fill(el::Number, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = apply_all(get_base(T), i -> el)
@inline Base.fill(f::Function, S::Type{T}) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = apply_all(get_base(T), i -> f())

# Array with zero/ones
@inline Base.zeros(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = fill(zero(T), dims)
@inline Base.ones(::Type{T}, dims::Int...) where {T <: Union{Tensor, SymmetricTensor, MixedTensor}} = fill(one(T), dims)

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
@inline basevec(::Type{Vec{dim, T}}) where {dim, T} = ntuple(i -> basevec(Vec{dim, T}, i), Val(dim))

@inline basevec(::Type{Vec{dim}}) where {dim} = basevec(Vec{dim, Float64})
@inline basevec(::Type{Vec{dim, T}}, i::Int) where {dim, T} = Vec{dim, T}(ntuple(j -> ifelse(j == i, one(T), zero(T)), Val(dim)))
@inline basevec(::Type{Vec{dim}}, i::Int) where {dim} = basevec(Vec{dim, Float64}, i)
@inline basevec(v::Vec{dim, T}) where {dim, T} = basevec(typeof(v))
@inline basevec(v::Vec{dim, T}, i::Int) where {dim, T} = basevec(typeof(v), i)

const eᵢ = basevec
