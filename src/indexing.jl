############
# Indexing #
############
@inline function compute_index(::Type{Tensor{1, dim}}, i::Int) where dim
    return i
end

@inline function compute_index(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int) where {dim}
    if i < j
        i, j = j, i
    end
    # We are skipping triangle over the diagonal = (j-1) * j / 2 indices
    skipped_indicies = div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@inline function compute_index(::Type{Tensor{2, dim}}, i::Int, j::Int) where {dim}
    return dim*(j-1) + i
end

@inline function compute_index(::Type{Tensor{3, dim}}, i::Int, j::Int, k::Int) where {dim}
    lower_order = Tensor{2, dim}
    I = compute_index(lower_order, i, j)
    n = n_components(lower_order)
    return (k-1) * n + I
end

@inline function compute_index(::Type{Tensor{4, dim}}, i::Int, j::Int, k::Int, l::Int) where {dim}
    lower_order = Tensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J-1) * n + I
end

@inline function compute_index(::Type{SymmetricTensor{4, dim}}, i::Int, j::Int, k::Int, l::Int) where {dim}
    lower_order = SymmetricTensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J-1) * n + I
end

# indexed with [order][dim]
const SYMMETRIC_INDICES = ((), ([1,], [1, 2, 4], [1, 2, 3, 5, 6, 9]), (),
                          ([1,], [1, 2, 4, 5, 6, 8, 13, 14, 16], [ 1,  2,
                            3,  5,  6,  9, 10, 11, 12, 14, 15, 18, 19, 20,
                            21, 23, 24, 27, 37, 38, 39, 41, 42, 45, 46, 47,
                            48, 50, 51, 54, 73, 74, 75, 77, 78, 81]))

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

# Slice
@inline Base.getindex(v::Vec, ::Colon) = v

function Base.getindex(S::Union{SecondOrderTensor, Tensor{3}, FourthOrderTensor}, ::Colon)
    throw(ArgumentError("S[:] not defined for S of order 2, 3, or 4, use Array(S) to convert to an Array"))
end

@inline @generated function Base.getindex(S::SecondOrderTensor{dim}, ::Colon, j::Int) where {dim}
    idx2(i,j) = compute_index(get_base(S), i, j)
    ex1 = Expr(:tuple, [:(get_data(S)[$(idx2(i,1))]) for i in 1:dim]...)
    ex2 = Expr(:tuple, [:(get_data(S)[$(idx2(i,2))]) for i in 1:dim]...)
    ex3 = Expr(:tuple, [:(get_data(S)[$(idx2(i,3))]) for i in 1:dim]...)
    return quote
        @boundscheck checkbounds(S,Colon(),j)
        if     j == 1 return Vec{dim}($ex1)
        elseif j == 2 return Vec{dim}($ex2)
        else          return Vec{dim}($ex3)
        end
    end
end
@inline @generated function Base.getindex(S::SecondOrderTensor{dim}, i::Int, ::Colon) where {dim}
    idx2(i,j) = compute_index(get_base(S), i, j)
    ex1 = Expr(:tuple, [:(get_data(S)[$(idx2(1,j))]) for j in 1:dim]...)
    ex2 = Expr(:tuple, [:(get_data(S)[$(idx2(2,j))]) for j in 1:dim]...)
    ex3 = Expr(:tuple, [:(get_data(S)[$(idx2(3,j))]) for j in 1:dim]...)
    return quote
        @boundscheck checkbounds(S,i,Colon())
        if     i == 1 return Vec{dim}($ex1)
        elseif i == 2 return Vec{dim}($ex2)
        else          return Vec{dim}($ex3)
        end
    end
end
