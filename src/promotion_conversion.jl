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
# NOTE: Base's contract for one-argument `promote(x)` is to return `(x,)`;
# it is not extended here (use `densify` to convert a symmetric tensor to a
# full one)
"""
    densify(S::AbstractTensor)

Convert a `SymmetricTensor` to the equivalent full `Tensor`; regular tensors
are returned unchanged.
"""
@inline densify(S1::AbstractTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, S1)

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

# Same tensor type, change element type (also Tensor -> Tensor, SymmetricTensor -> SymmetricTensor, MixedTensor -> MixedTensor)
@inline function Base.convert(::Type{TT}, x::AbstractTensor{order, dim}) where {order, dim, T, TT <: AbstractTensor{order, dim, T}}
    apply_all(get_base(TT), @inline function(i) @inbounds T(x.data[i]); end)
end

# Tensor -> SymmetricTensor
@inline function Base.convert(::Type{TT}, x::Tensor{order, dim}) where {order, dim, T, TT <: SymmetricTensor{order, dim, T}}
    if issymmetric(x)
        return convert(TT, symmetric(x))
    else
        throw(InexactError(:convert, TT, x))
    end
end

# SymmetricTensor -> Tensor
@inline function Base.convert(::Type{TT}, x::SymmetricTensor{order, dim}) where {order, dim, T, TT <: Tensor{order, dim, T}}
    Tensor{order, dim}(@inline function(args::Vararg{Int}) @inbounds T(x[args...]); end)
end

# Not passing eltype to conversion
@inline Base.convert(::Type{Tensor{order, dim}}, x::AbstractTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, x)
@inline Base.convert(::Type{SymmetricTensor{order, dim}}, x::AbstractTensor{order, dim, T}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, x)
@inline Base.convert(::Type{MixedTensor{order, dims}}, x::MixedTensor{order, dims, T}) where {order, dims, T} = convert(MixedTensor{order, dims, T}, x)
@inline Base.convert(::Type{Tensor}, x::SymmetricTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, x)
@inline Base.convert(::Type{SymmetricTensor}, x::Tensor{order, dim, T}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, x)
