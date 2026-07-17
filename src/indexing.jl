############
# Indexing #
############
# Map a Cartesian index to the linear index into the stored data tuple.
# Non-symmetric tensors store all components in column-major order.
@inline function compute_index(::Type{TT}, I::Vararg{Int, order}) where {order, TT <: Union{Tensor{order}, MixedTensor{order}}}
    return LinearIndices(size(TT))[I...]
end

@inline function compute_index(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int) where {dim}
    if i < j
        i, j = j, i
    end
    # We are skipping triangle over the diagonal = (j-1) * j / 2 indices
    skipped_indices = div((j - 1) * j, 2)
    return dim * (j - 1) + i - skipped_indices
end

@inline function compute_index(::Type{SymmetricTensor{4, dim}}, i::Int, j::Int, k::Int, l::Int) where {dim}
    lower_order = SymmetricTensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J - 1) * n + I
end

###################################################
# Component enumeration (single source of truth)  #
###################################################
# The independent components of a tensor type, in storage order, as Cartesian tuples.
function base_components(::Type{Tensor{order, dim}}) where {order, dim}
    return vec(collect(Iterators.product(ntuple(_ -> 1:dim, order)...)))
end
function base_components(::Type{SymmetricTensor{2, dim}}) where {dim}
    return [(i, j) for j in 1:dim for i in j:dim]
end
function base_components(::Type{SymmetricTensor{4, dim}}) where {dim}
    c2 = base_components(SymmetricTensor{2, dim})
    return [(i, j, k, l) for (k, l) in c2 for (i, j) in c2]
end
function base_components(::Type{TT}) where {order, TT <: MixedTensor{order}}
    return vec(collect(Iterators.product(map(d -> 1:d, size(TT))...)))
end

# Linear indices (into the data tuple of the corresponding full `Tensor`) of the
# independent components of a symmetric tensor. Replaces the old hardcoded
# SYMMETRIC_INDICES table.
function symmetric_indices(order::Int, dim::Int)
    return Int[compute_index(Tensor{order, dim}, c...) for c in base_components(SymmetricTensor{order, dim})]
end

###########################
# getindex general tensor #
###########################
@inline function Base.getindex(S::Tensor, i::Int)
    @boundscheck checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

@inline function Base.getindex(S::SymmetricTensor{2, dim}, i::Int, j::Int) where {dim}
    @boundscheck checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(SymmetricTensor{2, dim}, i, j)]
    return v
end

@inline function Base.getindex(S::SymmetricTensor{4, dim}, i::Int, j::Int, k::Int, l::Int) where {dim}
    @boundscheck checkbounds(S, i, j, k, l)
    @inbounds v = get_data(S)[compute_index(SymmetricTensor{4, dim}, i, j, k, l)]
    return v
end

@inline function Base.getindex(S::MixedTensor, i::Int)
    @boundscheck checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

# Slice
@inline Base.getindex(v::Vec, ::Colon) = v

function Base.getindex(S::Union{SecondOrderTensor, Tensor{3}, MixedTensor{3}, FourthOrderTensor}, ::Colon)
    throw(ArgumentError("S[:] not defined for S of order 2, 3, or 4, use Array(S) to convert to an Array"))
end

@inline @generated function Base.getindex(S::SecondOrderTensor, ::Colon, j::Int)
    dim1, dim2 = size(S)
    idx2(i, j) = compute_index(get_base(S), i, j)
    column(j) = Expr(:tuple, [:(get_data(S)[$(idx2(i, j))]) for i in 1:dim1]...)
    body = foldr(1:dim2; init = :(throw(BoundsError(S, (Colon(), j))))) do jc, rest
        :(j == $jc ? Vec{$dim1}($(column(jc))) : $rest)
    end
    return quote
        @boundscheck checkbounds(S, Colon(), j)
        @inbounds return $body
    end
end

@inline @generated function Base.getindex(S::SecondOrderTensor, i::Int, ::Colon)
    dim1, dim2 = size(S)
    idx2(i, j) = compute_index(get_base(S), i, j)
    row(i) = Expr(:tuple, [:(get_data(S)[$(idx2(i, j))]) for j in 1:dim2]...)
    body = foldr(1:dim1; init = :(throw(BoundsError(S, (i, Colon()))))) do ir, rest
        :(i == $ir ? Vec{$dim2}($(row(ir))) : $rest)
    end
    return quote
        @boundscheck checkbounds(S, i, Colon())
        @inbounds return $body
    end
end
