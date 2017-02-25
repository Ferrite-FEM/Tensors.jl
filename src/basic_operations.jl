###############
# Simple Math #
###############

# Unary
@inline Base.:+{T <: AbstractTensor}(S::T) = S
@inline Base.:-{T <: AbstractTensor}(S::T) = map(-, S)

# Binary
@inline Base.:+(S1::AbstractTensor, S2::AbstractTensor) = map(+, S1, S2)
@inline Base.:-(S1::AbstractTensor, S2::AbstractTensor) = map(-, S1, S2)
@inline Base.:*(S::AbstractTensor, n::Number) = map(x->(x*n), S)
@inline Base.:*(n::Number, S::AbstractTensor) = map(x->(n*x), S)
@inline Base.:/(S::AbstractTensor, n::Number) = map(x->(x/n), S)

# map implementations
@generated function Base.map{T <: AbstractTensor}(f, S::T)
    TensorType = get_base(S)
    N = n_components(TensorType)
    expr = Expr(:tuple, [:(f(get_data(S)[$i])) for i in 1:N]...)
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($expr)
    end
end

@inline Base.map{order, dim, T1, T2}(f, S1::AbstractTensor{order, dim, T1}, S2::AbstractTensor{order, dim, T2}) = ((SS1, SS2) = promote(S1, S2); map(f, SS1, SS2))
@generated function Base.map{T <: AllTensors}(f, S1::T, S2::T)
    TensorType = get_base(S1)
    N = n_components(TensorType)
    expr = Expr(:tuple, [:(f(get_data(S1)[$i], get_data(S2)[$i])) for i in 1:N]...)
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($expr)
    end
end
