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

@inline Base.:-(S1::Tensor{order, dim}, S2::Tensor{order, dim}) where {order, dim} = _map(-, S1, S2)
@inline Base.:-(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) where {order, dim} = _map(-, S1, S2)
@inline Base.:-(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) where {order, dim} = ((SS1, SS2) = promote_base(S1, S2); SS1 - SS2)

@inline Base.:*(S::AbstractTensor, n::Number) = _map(x -> x*n, S)
@inline Base.:*(n::Number, S::AbstractTensor) = _map(x -> n*x, S)
@inline Base.:/(S::AbstractTensor, n::Number) = _map(x -> x/n, S)

Base.:+(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))
Base.:-(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))

# map implementations
@inline function _map(f, S::AbstractTensor)
    return apply_all(S, @inline function(i) @inbounds f(S.data[i]); end)
end

# the caller of 2 arg _map MUST guarantee that both arguments have
# the same base (Tensor{order, dim} / SymmetricTensor{order, dim}) but not necessarily the same eltype
@inline function _map(f, S1::AllTensors, S2::AllTensors)
    return apply_all(S1, @inline function(i) @inbounds f(S1.data[i], S2.data[i]); end)
end

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

@inline _powdot(S1::Tensor, S2::Tensor) = dot(S1, S2)
@generated function _powdot(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim}) where {dim}
    idxS1(i, j) = compute_index(get_base(S1), i, j)
    idxS2(i, j) = compute_index(get_base(S2), i, j)
    exps = Expr(:tuple)
    for j in 1:dim, i in j:dim
        ex1 = Expr[:(get_data(S1)[$(idxS1(i, k))]) for k in 1:dim]
        ex2 = Expr[:(get_data(S2)[$(idxS2(k, j))]) for k in 1:dim]
        push!(exps.args, reducer(ex1, ex2))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return SymmetricTensor{2, dim}($exps)
    end
end

Base.iszero(a::AbstractTensor) = all(iszero, get_data(a))