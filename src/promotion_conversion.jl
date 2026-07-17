#############
# Promotion #
#############

# Promotion between two tensors promotes the eltype and promotes
# symmetric tensors to tensors

for TT in (Tensor, SymmetricTensor, MixedTensor)
    @eval @inline function Base.promote_rule(::Type{$TT{order, dim, A, M}},
                                             ::Type{$TT{order, dim, B, M}}) where {order, dim, A, B, M}
        $TT{order, dim, promote_type(A, B), M}
    end
end

# one direction suffices: Base tries promote_rule with both argument orders
@inline function Base.promote_rule(::Type{SymmetricTensor{order, dim, A, M1}},
                                   ::Type{Tensor{order, dim, B, M2}}) where {dim, A, B, order, M1, M2}
    Tensor{order, dim, promote_type(A, B), M2}
end

# inlined promote (promote in Base is not inlined)
@inline function Base.promote(S1::T, S2::S) where {T <: AbstractTensor, S <: AbstractTensor}
    TS = promote_type(T, S)
    return convert(TS, S1), convert(TS, S2)
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

# Identity, eltype change, and peeling off M (so that convert(typeof(...), ...)
# works) — identical for all three tensor types
for TT in (Tensor, SymmetricTensor, MixedTensor)
    @eval begin
        @inline Base.convert(::Type{$TT{order, dim, T}}, t::$TT{order, dim, T}) where {order, dim, T} = t
        @inline Base.convert(::Type{$TT{order, dim, T, M}}, t::$TT{order, dim, T, M}) where {order, dim, T, M} = t
        @inline function Base.convert(::Type{$TT{order, dim, T1}}, t::$TT{order, dim, T2}) where {order, dim, T1, T2}
            apply_all($TT{order, dim}, @inline function(i) @inbounds T1(t.data[i]); end)
        end
        @inline Base.convert(::Type{$TT{order, dim, T1, M}}, t::$TT{order, dim}) where {order, dim, T1, M} = convert($TT{order, dim, T1}, t)
    end
end

# Tensor <-> SymmetricTensor: peel off M and fill in the eltype from the source
@inline Base.convert(::Type{Tensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim}) where {order, dim, T1, M} = convert(Tensor{order, dim, T1}, t)
@inline Base.convert(::Type{SymmetricTensor{order, dim, T1, M}}, t::Tensor{order, dim}) where {order, dim, T1, M} = convert(SymmetricTensor{order, dim, T1}, t)

@inline Base.convert(::Type{Tensor{order, dim}}, t::SymmetricTensor{order, dim, T}) where {order, dim, T} = convert(Tensor{order, dim, T}, t)
@inline Base.convert(::Type{SymmetricTensor{order, dim}}, t::Tensor{order, dim, T}) where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, t)
@inline Base.convert(::Type{Tensor}, t::SymmetricTensor{order, dim, T})             where {order, dim, T} = convert(Tensor{order, dim, T}, t)
@inline Base.convert(::Type{SymmetricTensor}, t::Tensor{order, dim, T})             where {order, dim, T} = convert(SymmetricTensor{order, dim, T}, t)

# SymmetricTensor -> Tensor
@inline function Base.convert(::Type{Tensor{order, dim, T1}}, t::SymmetricTensor{order, dim}) where {order, dim, T1}
    Tensor{order, dim}(@inline function(inds...) @inbounds T1(t[inds...]); end)
end

# Tensor -> SymmetricTensor
@inline function Base.convert(::Type{SymmetricTensor{order, dim, T1}}, t::Tensor{order, dim}) where {dim, order, T1}
    if issymmetric(t)
        return convert(SymmetricTensor{order, dim, T1}, symmetric(t))
    else
        throw(InexactError(:convert, SymmetricTensor{order, dim, T1}, t))
    end
end
