###############
# Simple Math #
###############

# Unary
@inline Base.:+{T <: AbstractTensor}(S::T) = S
@inline Base.:-{T <: AbstractTensor}(S::T) = _map(-, S)

# Binary
@inline Base.:+{order, dim}(S1::Tensor{order, dim}, S2::Tensor{order, dim}) = _map(+, S1, S2)
@inline Base.:+{order, dim}(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) = _map(+, S1, S2)
@inline Base.:-{order, dim}(S1::Tensor{order, dim}, S2::Tensor{order, dim}) = _map(-, S1, S2)
@inline Base.:-{order, dim}(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim}) = _map(-, S1, S2)
@inline Base.:-{order, dim}(S1::AnyTensor{order, dim}, S2::AnyTensor{order, dim}) =  ((SS1, SS2) = promote(S1, S2); _map(-, SS1, SS2))
@inline Base.:+{order, dim}(S1::AnyTensor{order, dim}, S2::AnyTensor{order, dim}) =  ((SS1, SS2) = promote(S1, S2); _map(+, SS1, SS2))

@inline Base.:*(S::AbstractTensor, n::Number) = _map(x -> x*n, S)
@inline Base.:*(n::Number, S::AbstractTensor) = _map(x -> n*x, S)
@inline Base.:/(S::AbstractTensor, n::Number) = _map(x -> x/n, S)

Base.:+(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))
Base.:-(S1::AbstractTensor, S2::AbstractTensor) = throw(DimensionMismatch("dimension and order must match"))

# map implementations
@inline function _map{T <: AbstractTensor}(f, S::T)
    return apply_all(S, @inline function(i) @inboundsret f(S.data[i]); end)
end

_map{order, dim}(f, S1::Tensor{order, dim}, S2::SymmetricTensor{order, dim}) = ((SS1, SS2) = promote(S1, S2); _map(f, SS1, SS2))
_map{order, dim}(f, S1::SymmetricTensor{order, dim}, S2::Tensor{order, dim}) = ((SS1, SS2) = promote(S1, S2); _map(f, SS1, SS2))

@inline function _map(f, S1::AllTensors, S2::AllTensors)
    return apply_all(S1, @inline function(i) @inboundsret f(S1.data[i], S2.data[i]); end)
end
