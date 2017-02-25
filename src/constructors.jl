# zero, one, rand
for (op, el) in ((:zero, :(zero(T))), (:ones, :(one(T))), (:rand, :(()->rand(T))))
for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @inline Base.$op{order, dim}(S::Type{$TensorType{order, dim}}) = $op($TensorType{order, dim, Float64})
        @inline Base.$op{order, dim, T, N}(S::Type{$TensorType{order, dim, T, N}}) = $op($TensorType{order, dim, T})
        @inline Base.$op{order, dim, T}(S::Type{$TensorType{order, dim, T}}) = fill($el, $TensorType{order, dim})
    end
end
@eval @inline Base.$op{dim}(S::Type{Vec{dim}}) = $op(Vec{dim, Float64})
@eval @inline Base.$op(t::AllTensors) = $op(typeof(t))
end

@generated function Base.fill{T <: AbstractTensor}(el::Union{Number, Function}, S::Type{T})
    TensorType = get_base(get_type(S))
    N = n_components(TensorType)
    expr = Expr(:tuple)
    ele = el <: Number ? :(el) : :(el())
    for i in 1:N
        push!(expr.args, ele)
    end
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
    f = (i,j) -> i == j ? :($ET(v[$i])) : :($(zero(ET)))
    exp = tensor_create(TensorType, f)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end
@inline Base.diagm{dim, T <: Number}(::Type{Tensor{2, dim}}, v::T) = v * one(Tensor{2, dim, T})
@inline Base.diagm{dim, T <: Number}(::Type{SymmetricTensor{2, dim}}, v::T) = v * one(SymmetricTensor{2, dim, T})