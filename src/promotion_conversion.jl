#############
# Promotion #
#############

# Promotion between two tensors promote the eltype and promotes
# symmetric tensors to tensors

@inline function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{SymmetricTensor{order, dim, A, M}},
                                                                     ::Type{SymmetricTensor{order, dim, B, M}})
    SymmetricTensor{order, dim, promote_type(A, B), M}
end

@inline function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{Tensor{order, dim, A, M}},
                                                                     ::Type{Tensor{order, dim, B, M}})
    Tensor{order, dim, promote_type(A, B), M}
end

@inline function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{SymmetricTensor{order, dim, A, M1}},
                                                                          ::Type{Tensor{order, dim, B, M2}})
    Tensor{order, dim, promote_type(A, B), M2}
end

@inline function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{Tensor{order, dim, A, M1}},
                                                                          ::Type{SymmetricTensor{order, dim, B, M2}})
    Tensor{order, dim, promote_type(A, B), M1}
end

# define a base promotion that only promotes SymmetricTensor to Tensor but leaves eltype
@inline function promote_base{order, dim}(S1::Tensor{order, dim}, S2::SymmetricTensor{order, dim})
    return S1, convert(Tensor{order, dim}, S2)
end
@inline function promote_base{order, dim}(S1::SymmetricTensor{order, dim}, S2::Tensor{order, dim})
    return convert(Tensor{order, dim}, S1), S2
end

###############
# Conversions #
###############

# Identity conversions
@inline Base.convert{order, dim, T}(::Type{Tensor{order, dim, T}}, t::Tensor{order, dim, T}) = t
@inline Base.convert{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}, t::Tensor{order, dim, T, M}) = t
@inline Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, t::SymmetricTensor{order, dim, T}) = t
@inline Base.convert{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}, t::SymmetricTensor{order, dim, T, M}) = t

# Change element type
@inline function Base.convert{order, dim, T1, T2}(::Type{Tensor{order, dim, T1}}, t::Tensor{order, dim, T2})
    apply_all(Tensor{order, dim}, @inline function(i) @inboundsret T1(t.data[i]); end)
end

@inline function Base.convert{order, dim, T1, T2}(::Type{SymmetricTensor{order, dim, T1}}, t::SymmetricTensor{order, dim, T2})
    apply_all(SymmetricTensor{order, dim}, @inline function(i) @inboundsret T1(t.data[i]); end)
end

# Peel off the M but define these so that convert(typeof(...), ...) works
@inline Base.convert{order, dim, T1, M}(::Type{Tensor{order, dim, T1, M}}, t::Tensor{order, dim}) = convert(Tensor{order, dim, T1}, t)
@inline Base.convert{order, dim, T1, M}(::Type{SymmetricTensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim}) = convert(SymmetricTensor{order, dim, T1}, t)
@inline Base.convert{order, dim, T1, M}(::Type{Tensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim}) = convert(Tensor{order, dim, T1}, t)
@inline Base.convert{order, dim, T1, M}(::Type{SymmetricTensor{order, dim, T1, M}}, t::Tensor{order, dim}) = convert(SymmetricTensor{order, dim, T1}, t)

@inline Base.convert{order, dim, T}(::Type{Tensor{order, dim}}, t::SymmetricTensor{order, dim, T}) = convert(Tensor{order, dim, T}, t)
@inline Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim}}, t::Tensor{order, dim, T}) = convert(SymmetricTensor{order, dim, T}, t)
@inline Base.convert{order, dim, T}(::Type{Tensor}, t::SymmetricTensor{order, dim, T}) = convert(Tensor{order, dim, T}, t)
@inline Base.convert{order, dim, T}(::Type{SymmetricTensor}, t::Tensor{order, dim, T}) = convert(SymmetricTensor{order, dim, T}, t)

# SymmetricTensor -> Tensor
function Base.convert{dim, T1, T2}(::Type{Tensor{2, dim, T1}}, t::SymmetricTensor{2, dim, T2})
    Tensor{2, dim}(@inline function(i,j) @inboundsret T1(t[i,j]); end)
end

function Base.convert{dim, T1, T2}(::Type{Tensor{4, dim, T1}}, t::SymmetricTensor{4, dim, T2})
    Tensor{4, dim}(@inline function(i,j,k,l) @inboundsret T1(t[i,j,k,l]); end)
end


# Tensor -> SymmetricTensor
function Base.convert{dim, order, T1}(::Type{SymmetricTensor{order, dim, T1}}, t::Tensor{order, dim})
    if issymmetric(t)
        return convert(SymmetricTensor{order, dim, T1}, symmetric(t))
    else
        throw(InexactError())
    end
end
