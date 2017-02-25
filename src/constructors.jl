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

# Array with zero/ones
@inline Base.zeros{T <: AbstractTensor}(::Type{T}, dims...) = fill(zero(T), dims...)
@inline Base.ones{T <: AbstractTensor}(::Type{T}, dims...) = fill(one(T), dims...)

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
