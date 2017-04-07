###############
# Simple Math #
###############

# Unary
@inline Base.:+(S::AbstractTensor) = S
@inline Base.:-(S::AbstractTensor) = _map(-, S)

# Binary
@inline Base.:+{order, dim}(S1::Tensor{order, dim}, S2::Tensor{order, dim}) = _map(+, S1, S2)
@inline Base.:+{order, dim}(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) = _map(+, S1, S2)
@inline Base.:+{order, dim}(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) = ((SS1, SS2) = promote_base(S1, S2); SS1 + SS2)

@inline Base.:-{order, dim}(S1::Tensor{order, dim}, S2::Tensor{order, dim}) = _map(-, S1, S2)
@inline Base.:-{order, dim}(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) = _map(-, S1, S2)
@inline Base.:-{order, dim}(S1::AbstractTensor{order, dim}, S2::AbstractTensor{order, dim}) = ((SS1, SS2) = promote_base(S1, S2); SS1 - SS2)

@inline Base.:*(S::AbstractTensor, n::Number) = _map(x -> x*n, S)
@inline Base.:*(n::Number, S::AbstractTensor) = _map(x -> n*x, S)
@inline Base.:/(S::AbstractTensor, n::Number) = _map(x -> x/n, S)
function Base.:^(S::SecondOrderTensor, p::Int)
    if p == 1
        return S
    elseif p == 0
        return one(S)
    elseif p == -1
        return inv(S)
    elseif p < 0
        throw(DomainError())
    end
    t = S
    for _ in 2:p
        t = t ⋅ S
    end
    return t
end

Base.:+(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))
Base.:-(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))

# map implementations
@inline function _map(f, S::AbstractTensor)
    return apply_all(S, @inline function(i) @inboundsret f(S.data[i]); end)
end

# the caller of 2 arg _map MUST guarantee that both arguments have
# the same base (Tensor{order, dim} / SymmetricTensor{order, dim}) but not necessarily the same eltype
@inline function _map(f, S1::AllTensors, S2::AllTensors)
    return apply_all(S1, @inline function(i) @inboundsret f(S1.data[i], S2.data[i]); end)
end
