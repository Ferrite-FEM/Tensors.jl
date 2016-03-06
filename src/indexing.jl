
# Indexing #
############
@inline function compute_index{dim}(::Type{SymmetricTensor{2, dim}}, i::Int, j::Int)
    if i < j
        i, j  = j,i
    end
    # We are skipping triangle over the diagonal = (j-1) * j / 2 indices
    skipped_indicies = div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@inline function compute_index{dim}(::Type{Tensor{2, dim}}, i::Int, j::Int)
    return dim*(j-1) + i
end

#@inline function compute_index{dim}(::Type{SymmetricTensor{4, dim}},
#                                    i::Int, j::Int, k::Int, l::Int)
#    lower_order = SymmetricTensor{2,dim}
#    I = compute_index(lower_order, i, j)
#    J = compute_index(lower_order, k, l)
#    n = n_components(lower_order)
#    return (J-1) * n + I
#end


@inline function compute_index{dim}(::Union{Type{SymmetricTensor{4, dim}}, Type{Tensor{4, dim}}},
                                    i::Int, j::Int, k::Int, l::Int)
    lower_order = SymmetricTensor{2,dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    n = n_components(lower_order)
    return (J-1) * n + I
end


# getindex general tensor #
############
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
    @inbounds v = S.data[compute_index(SymmetricTensor{4, dim}, i, j, k, l)]
    return v
end



# setindex #
#############

@inline function setindex{dim}(S::Tensor{1, dim}, v, i::Int)
    @boundscheck checkbounds(S, i)
    t = typeof(S)(vec_set_index(S.data, v, Val{i}))
    return t
end

@inline function setindex{dim}(S::Tensor{2, dim}, v, i::Int, j::Int)
    @boundscheck checkbounds(S, i, j)
    t = typeof(S)(mat_set_index(S.data, v, Val{i}, Val{j}))
    return t
end

@inline function setindex{dim}(S::Tensor{4, dim}, v, i::Int, j::Int, k::Int, l::Int)
    @boundscheck checkbounds(S, i, j)
    lower_order = Tensor{2,dim}
    I = compute_index(lower_order, i, j)
    J = compute_index(lower_order, k, l)
    t = typeof(S)(mat_set_index(S.data, v, Val{I}, Val{J}))
    return t
end


@inline function setindex{dim}(S::SymmetricTensor{2, dim}, v, i::Int, j::Int)
    @boundscheck checkbounds(S, i, j)
    if i < j
        i, j  = j,i
    end
    t = typeof(S)(sym_mat_set_index(S.data, v, Val{i}, Val{j}))
    return t
end
#
#@inline function setindex{dim}(S::SymmetricTensor{4, dim}, v, i::Int, j::Int, k::Int, l::Int)
#    @boundscheck checkbounds(S, i, j)
#    lower_order = Tensor{2,dim}
#    if i < j
#        i, j  = j,i
#    end
#    if k < l
#        k, l  = l,k
#    end
#    I = compute_index(lower_order, i, j)
#    J = compute_index(lower_order, k, l)
#    t = typeof(S)(sym_mat_set_index(S.data, v, Val{I}, Val{J}))
#    return t
#end

