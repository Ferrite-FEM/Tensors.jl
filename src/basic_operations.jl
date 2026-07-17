###############
# Simple Math #
###############

# Unary
@inline Base.:+(S::AbstractTensor) = S
@inline Base.:-(S::AbstractTensor) = _map(-, S)

# Binary
for op in (:+, :-)
    @eval begin
        @inline Base.$op(S1::Tensor{order, dim}, S2::Tensor{order, dim}) where {order, dim} = _map($op, S1, S2)
        @inline Base.$op(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) where {order, dim} = _map($op, S1, S2)
        @inline Base.$op(S1::MixedTensor{order, dims}, S2::MixedTensor{order, dims}) where {order, dims} = _map($op, S1, S2)
        # Tensor with SymmetricTensor: promote the symmetric one and retry
        @inline Base.$op(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) where {order, dim} = ((SS1, SS2) = promote_base(S1, S2); $op(SS1, SS2))
        Base.$op(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))
    end
end

@inline Base.:*(S::AbstractTensor, n::Number) = _map(x -> x * n, S)
@inline Base.:*(n::Number, S::AbstractTensor) = _map(x -> n * x, S)
@inline Base.:/(S::AbstractTensor, n::Number) = _map(x -> x / n, S)

@inline Base.literal_pow(::typeof(^), S::SecondOrderTensor, ::Val{p}) where {p} = S^p
@inline function Base.literal_pow(::typeof(^), S::SecondOrderTensor, ::Val{-1})
    p = -1
    return S^p
end

# non-literal integer powers (issue #239). For small exponents this is the
# same repeated-contraction evaluation as `literal_pow`, so `S^3` and
# `S^(p::Int)` with `p = 3` give identical results; larger exponents use
# exponentiation by squaring.
@inline function Base.:^(S1::SecondOrderTensor, p::Integer)
    p == 0 && return one(S1)
    # checked: -typemin overflows silently and would recurse forever
    p < 0 && return inv(S1)^(Base.checked_neg(p))
    if p <= 4
        S2 = S1
        for i in 2:p
            S2 = _powdot(S2, S1)
        end
        return S2
    end
    # exponentiation by squaring
    t = trailing_zeros(p) + 1
    p >>= t
    S2 = S1
    while (t -= 1) > 0
        S2 = _powdot(S2, S2)
    end
    y = S2
    while p > 0
        t = trailing_zeros(p) + 1
        p >>= t
        while (t -= 1) >= 0
            S2 = _powdot(S2, S2)
        end
        y = _powdot(y, S2)
    end
    return y
end
Base.:^(S::MixedTensor2, p::Integer) = throw(ArgumentError("The exponentiation, S^$p, is not defined for S::MixedTensor2"))

@inline _powdot(S1::Tensor, S2::Tensor) = dot(S1, S2)
# Powers of a symmetric tensor commute, so the single contraction stays symmetric.
@tensorop function _powdot(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim}) where {dim}
    C[i, j]::SymmetricTensor{2, dim} = S1[i, k] * S2[k, j]
end

Base.iszero(a::AbstractTensor) = all(iszero, a.data)
