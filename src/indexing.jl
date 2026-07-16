############
# Indexing #
############
# Map a Cartesian index to the linear index into the stored data tuple.
@inline function compute_index(::Type{Tensor{1, dim}}, i::Int) where {dim}
    return i
end

@inline function compute_index(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int) where {dim}
    if i < j
        i, j = j, i
    end
    # We are skipping triangle over the diagonal = (j-1) * j / 2 indices
    skipped_indices = div((j - 1) * j, 2)
    return dim * (j - 1) + i - skipped_indices
end

@inline function compute_index(::Type{Tensor{2, dim}}, i::Int, j::Int) where {dim}
    return dim * (j - 1) + i
end

@inline function compute_index(::Type{Tensor{3, dim}}, i::Int, j::Int, k::Int) where {dim}
    lower_order = Tensor{2, dim}
    I = compute_index(lower_order, i, j)
    n = n_components(lower_order)
    return (k - 1) * n + I
end

@inline function compute_index(::Type{Tensor{4, dim}}, i::Int, j::Int, k::Int, l::Int) where {dim}
    lower_order = Tensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J - 1) * n + I
end

@inline function compute_index(::Type{SymmetricTensor{4, dim}}, i::Int, j::Int, k::Int, l::Int) where {dim}
    lower_order = SymmetricTensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J - 1) * n + I
end

# MixedTensor
@inline compute_index(::Type{<:MixedTensor{1}}, i::Int) = i
@inline function compute_index(::Type{<:MixedTensor2{dim1}}, i::Int, j::Int) where {dim1}
    return (j - 1) * dim1 + i
end
@inline function compute_index(::Type{<:MixedTensor3{dim1, dim2}}, i::Int, j::Int, k::Int) where {dim1, dim2}
    return (k - 1) * (dim2 * dim1) + (j - 1) * dim1 + i
end
@inline function compute_index(::Type{<:MixedTensor4{dim1, dim2, dim3}}, i::Int, j::Int, k::Int, l::Int) where {dim1, dim2, dim3}
    n3, n2, n1 = (dim3 * dim2, dim2, 1) .* dim1
    return (l - 1) * n3 + (k - 1) * n2 + (j - 1) * n1 + i
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

function Base.getindex(S::Union{SecondOrderTensor, Tensor{3}, FourthOrderTensor}, ::Colon)
    throw(ArgumentError("S[:] not defined for S of order 2, 3, or 4, use Array(S) to convert to an Array"))
end

@inline @generated function Base.getindex(S::SecondOrderTensor, ::Colon, j::Int)
    dim1, dim2 = size(S)
    idx2(i, j) = compute_index(get_base(S), i, j)
    exprs = [Expr(:tuple, [:(get_data(S)[$(idx2(i, j))]) for i in 1:dim1]...) for j in 1:dim2]
    branches = :(throw(BoundsError(S, (Colon(), j))))
    for j in dim2:-1:1
        branches = Expr(:elseif, :(j == $j), :(return Vec{$dim1}($(exprs[j]))), branches)
    end
    branches = Expr(:if, branches.args...)
    return quote
        @boundscheck checkbounds(S, Colon(), j)
        @inbounds $branches
    end
end

@inline @generated function Base.getindex(S::SecondOrderTensor, i::Int, ::Colon)
    dim1, dim2 = size(S)
    idx2(i, j) = compute_index(get_base(S), i, j)
    exprs = [Expr(:tuple, [:(get_data(S)[$(idx2(i, j))]) for j in 1:dim2]...) for i in 1:dim1]
    branches = :(throw(BoundsError(S, (i, Colon()))))
    for i in dim1:-1:1
        branches = Expr(:elseif, :(i == $i), :(return Vec{$dim2}($(exprs[i]))), branches)
    end
    branches = Expr(:if, branches.args...)
    return quote
        @boundscheck checkbounds(S, i, Colon())
        @inbounds $branches
    end
end
