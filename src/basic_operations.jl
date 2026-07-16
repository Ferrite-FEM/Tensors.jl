###############
# Simple Math #
###############

# Unary
@inline Base.:+(S::AbstractTensor) = S
@inline Base.:-(S::AbstractTensor) = _map(-, S)

# Binary
@inline Base.:+(S1::Tensor{order, dim}, S2::Tensor{order, dim}) where {order, dim} = _map(+, S1, S2)
@inline Base.:+(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) where {order, dim} = _map(+, S1, S2)
@inline Base.:+(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) where {order, dim} = ((SS1, SS2) = promote_base(S1, S2); SS1 + SS2)
@inline Base.:+(S1::MixedTensor{order, dims}, S2::MixedTensor{order, dims}) where {order, dims} = _map(+, S1, S2)

@inline Base.:-(S1::Tensor{order, dim}, S2::Tensor{order, dim}) where {order, dim} = _map(-, S1, S2)
@inline Base.:-(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) where {order, dim} = _map(-, S1, S2)
@inline Base.:-(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) where {order, dim} = ((SS1, SS2) = promote_base(S1, S2); SS1 - SS2)
@inline Base.:-(S1::MixedTensor{order, dims}, S2::MixedTensor{order, dims}) where {order, dims} = _map(-, S1, S2)

@inline Base.:*(S::AbstractTensor, n::Number) = _map(x -> x * n, S)
@inline Base.:*(n::Number, S::AbstractTensor) = _map(x -> n * x, S)
@inline Base.:/(S::AbstractTensor, n::Number) = _map(x -> x / n, S)

Base.:+(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))
Base.:-(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))

# power
@inline Base.literal_pow(::typeof(^), S::SecondOrderTensor, ::Val{-1}) = inv(S)
@inline Base.literal_pow(::typeof(^), S::SecondOrderTensor, ::Val{0})  = one(S)
@inline Base.literal_pow(::typeof(^), S::SecondOrderTensor, ::Val{1})  = S
@inline function Base.literal_pow(::typeof(^), S1::SecondOrderTensor, ::Val{p}) where {p}
    p > 0 ? (S2 = S1; q = p) : (S2 = inv(S1); q = -p)
    S3 = S2
    for i in 2:q
        S2 = _powdot(S2, S3)
    end
    S2
end

function Base.literal_pow(::typeof(^), S::MixedTensor2, ::Val{p}) where {p}
    throw(ArgumentError("The exponentiation, S^$p, is not defined for S::MixedTensor2"))
end

@inline _powdot(S1::Tensor, S2::Tensor) = dot(S1, S2)
# Powers of a symmetric tensor commute, so the single contraction stays symmetric.
@generated function _powdot(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim}) where {dim}
    expr = einsum_expr((:i, :j), IndexedArg(:S1, get_base(S1), (:i, :k)), IndexedArg(:S2, get_base(S2), (:k, :j));
                       force_out = SymmetricTensor{2, dim})
    return quote
        $(Expr(:meta, :inline))
        $expr
    end
end

Base.iszero(a::AbstractTensor) = all(iszero, get_data(a))
