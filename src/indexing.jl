############
# Indexing #
############

@inline function compute_index{dim}(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int)
    if i < j
        i, j = j, i
    end
    # We are skipping triangle over the diagonal = (j-1) * j / 2 indices
    skipped_indicies = div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@inline function compute_index{dim}(::Type{Tensor{2, dim}}, i::Int, j::Int)
    return dim*(j-1) + i
end

@inline function compute_index{dim}(T::Type{Tensor{4, dim}}, i::Int, j::Int, k::Int, l::Int)
    lower_order = Tensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J-1) * n + I
end

@inline function compute_index{dim}(T::Type{SymmetricTensor{4, dim}}, i::Int, j::Int, k::Int, l::Int)
    lower_order = SymmetricTensor{2, dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J-1) * n + I
end

###########################
# getindex general tensor #
###########################
@inline function Base.getindex(S::Tensor, i::Int)
    @boundscheck checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

@inline function Base.getindex{dim}(S::SymmetricTensor{2, dim}, i::Int, j::Int)
    @boundscheck checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(SymmetricTensor{2, dim}, i, j)]
    return v
end

@inline function Base.getindex{dim}(S::SymmetricTensor{4, dim}, i::Int, j::Int, k::Int, l::Int)
    @boundscheck checkbounds(S, i, j, k, l)
    @inbounds v = get_data(S)[compute_index(SymmetricTensor{4, dim}, i, j, k, l)]
    return v
end

# Slice
@inline Base.getindex(v::Vec, ::Colon) = v

function Base.getindex(S::Union{SecondOrderTensor, FourthOrderTensor}, ::Colon)
    throw(ArgumentError("S[:] not defined for S of order 2 or 4, use Array(S) to convert to an Array"))
end

@inline @generated function Base.getindex{dim, T}(S::SecondOrderTensor{dim, T}, ::Colon, j::Int)
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
@inline @generated function Base.getindex{dim, T}(S::SecondOrderTensor{dim, T}, i::Int, ::Colon)
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
