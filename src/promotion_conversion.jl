#############
# Promotion #
#############

# Promotion between two tensors promote the eltype and promotes
# symmetric tensors to tensors

@inline function Base.promote_rule(::Type{SymmetricTensor{order, dim, A, M}},
                                   ::Type{SymmetricTensor{order, dim, B, M}}) where {dim, A, B, order, M}
    SymmetricTensor{order, dim, promote_type(A, B), M}
end

@inline function Base.promote_rule(::Type{Tensor{order, dim, A, M}},
                                   ::Type{Tensor{order, dim, B, M}}) where {dim, A, B, order, M}
    Tensor{order, dim, promote_type(A, B), M}
end

@inline function Base.promote_rule(::Type{SymmetricTensor{order, dim, A, M1}},
                                   ::Type{Tensor{order, dim, B, M2}}) where {dim, A, B, order, M1, M2}
    Tensor{order, dim, promote_type(A, B), M2}
end

@inline function Base.promote_rule(::Type{Tensor{order, dim, A, M1}},
                                   ::Type{SymmetricTensor{order, dim, B, M2}}) where {dim, A, B, order, M1, M2}
    Tensor{order, dim, promote_type(A, B), M1}
end

# inlined promote (promote in Base is not inlined)
@inline function Base.promote(S1::T, S2::S) where {T <: AbstractTensor, S <: AbstractTensor}
    return convert(promote_type(T, S), S1), convert(promote_type(T, S), S2)
end
@inline Base.promote(S1::AbstractTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, S1)

# base promotion that only promotes SymmetricTensor to Tensor but leaves eltype
@inline function promote_base(S1::Tensor{order, dim}, S2::SymmetricTensor{order, dim}) where {order, dim}
    return S1, convert(Tensor{order, dim}, S2)
end
@inline function promote_base(S1::SymmetricTensor{order, dim}, S2::Tensor{order, dim}) where {order, dim}
    return convert(Tensor{order, dim}, S1), S2
end

###############
# Conversions #
###############

# Identity conversions
@inline Base.convert(::Type{Tensor{order, dim, T}}, t::Tensor{order, dim, T}) where {order, dim, T} = t
@inline Base.convert(::Type{Tensor{order, dim, T, M}}, t::Tensor{order, dim, T, M}) where {order, dim, T, M} = t
@inline Base.convert(::Type{SymmetricTensor{order, dim, T}}, t::SymmetricTensor{order, dim, T}) where {order, dim, T} = t
@inline Base.convert(::Type{SymmetricTensor{order, dim, T, M}}, t::SymmetricTensor{order, dim, T, M}) where {order, dim, T, M} = t

# Change element type
@inline function Base.convert(::Type{Tensor{order, dim, T1}}, t::Tensor{order, dim, T2}) where {order, dim, T1, T2}
    apply_all(Tensor{order, dim}, @inline function(i) @inbounds T1(t.data[i]); end)
end

@inline function Base.convert(::Type{SymmetricTensor{order, dim, T1}}, t::SymmetricTensor{order, dim, T2}) where {order, dim, T1, T2}
    apply_all(SymmetricTensor{order, dim}, @inline function(i) @inbounds T1(t.data[i]); end)
end

# Peel off the M but define these so that convert(typeof(...), ...) works
@inline Base.convert(::Type{Tensor{order, dim, T1, M}}, t::Tensor{order, dim})                   where {order, dim, T1, M} = convert(Tensor{order, dim, T1}, t)
@inline Base.convert(::Type{SymmetricTensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim}) where {order, dim, T1, M} = convert(SymmetricTensor{order, dim, T1}, t)
@inline Base.convert(::Type{Tensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim})          where {order, dim, T1, M} = convert(Tensor{order, dim, T1}, t)
@inline Base.convert(::Type{SymmetricTensor{order, dim, T1, M}}, t::Tensor{order, dim})          where {order, dim, T1, M} = convert(SymmetricTensor{order, dim, T1}, t)

@inline Base.convert(::Type{Tensor{order, dim}}, t::SymmetricTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, t)
@inline Base.convert(::Type{SymmetricTensor{order, dim}}, t::Tensor{order, dim, T}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, t)
@inline Base.convert(::Type{Tensor}, t::SymmetricTensor{order, dim, T})             where {order, dim, T} = convert(Tensor{order, dim, T}, t)
@inline Base.convert(::Type{SymmetricTensor}, t::Tensor{order, dim, T})             where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, t)

# SymmetricTensor -> Tensor
@inline function Base.convert(::Type{Tensor{2, dim, T1}}, t::SymmetricTensor{2, dim, T2}) where {dim, T1, T2}
    Tensor{2, dim}(@inline function(i,j) @inbounds T1(t[i,j]); end)
end

@inline function Base.convert(::Type{Tensor{4, dim, T1}}, t::SymmetricTensor{4, dim, T2}) where {dim, T1, T2}
    Tensor{4, dim}(@inline function(i,j,k,l) @inbounds T1(t[i,j,k,l]); end)
end

# Tensor -> SymmetricTensor
@inline function Base.convert(::Type{SymmetricTensor{order, dim, T1}}, t::Tensor{order, dim}) where {dim, order, T1}
    if issymmetric(t)
        return convert(SymmetricTensor{order, dim, T1}, symmetric(t))
    else
        throw(InexactError(:convert, SymmetricTensor{order, dim, T1}, t))
    end
end
