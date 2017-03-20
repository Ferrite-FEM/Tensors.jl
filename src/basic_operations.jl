###############
# Simple Math #
###############

# Unary
@inline Base.:+{T <: AbstractTensor}(S::T) = S
@inline Base.:-{T <: AbstractTensor}(S::T) = map(-, S)

# Binary
@inline Base.:+{order, dim, T}(S1::Tensor{order, dim, T}, S2::Tensor{order, dim, T}) = map(+, S1, S2)
@inline Base.:+{order, dim, T}(S1::SymmetricTensor{order, dim, T}, S2::SymmetricTensor{order, dim, T}) = map(+, S1, S2)
@inline Base.:+(S1::AbstractTensor, S2::AbstractTensor) = +(promote(S1, S2)...)
@inline Base.:-{order, dim, T}(S1::Tensor{order, dim, T}, S2::Tensor{order, dim, T}) = map(-, S1, S2)
@inline Base.:-{order, dim, T}(S1::SymmetricTensor{order, dim, T}, S2::SymmetricTensor{order, dim, T}) = map(-, S1, S2)
@inline Base.:-(S1::AbstractTensor, S2::AbstractTensor) = -(promote(S1, S2)...)

@inline Base.:*(S::AbstractTensor, n::Number) = map(x->(x*n), S)
@inline Base.:*(n::Number, S::AbstractTensor) = map(x->(n*x), S)
@inline Base.:/(S::AbstractTensor, n::Number) = map(x->(x/n), S)

# map implementations
@inline function Base.map{T <: AbstractTensor}(f, S::T)
    return apply_all(S, @inline function(i) @inboundsret f(S.data[i]); end)
end

@inline Base.map{order, dim, T1, T2}(f, S1::AbstractTensor{order, dim, T1}, S2::AbstractTensor{order, dim, T2}) = ((SS1, SS2) = promote(S1, S2); map(f, SS1, SS2))

@inline function Base.map{T <: AllTensors}(f, S1::T, S2::T)
    return apply_all(S1, @inline function(i) @inboundsret f(S1.data[i], S2.data[i]); end)
end
