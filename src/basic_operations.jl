###############
# Simple Math #
###############

# Unary
@inline Base.:+{order, dim}(S::AbstractTensor{order, dim}) = S
@inline Base.:-{order, dim}(S::AbstractTensor{order, dim}) = map(-, S)

# Binary
@inline Base.:+(S1::AbstractTensor, S2::AbstractTensor) = map(+, S1, S2)
@inline Base.:-(S1::AbstractTensor, S2::AbstractTensor) = map(-, S1, S2)
@inline Base.:*(S::AbstractTensor, n::Number) = map(x->(x*n), S)
@inline Base.:*(n::Number, S::AbstractTensor) = map(x->(n*x), S)
@inline Base.:/(S::AbstractTensor, n::Number) = map(x->(x/n), S)

# map implementations
Base.map{order, dim}(f, S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) = _map(f, promote(S1, S2)...)

@generated function Base.map{T<:AbstractTensor}(f, S::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    expr = Expr(:tuple)
    for i in 1:N
        push!(expr.args, :(f(get_data(S)[$i])))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($expr)
    end
end

@generated function _map{T<:AbstractTensor}(f, S1::T, S2::T)
    TensorType = get_base(S1)
    N = n_components(TensorType)
    expr = Expr(:tuple)
    for i in 1:N
        push!(expr.args, :(f(get_data(S1)[$i], get_data(S2)[$i])))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($expr)
    end
end
