# Convert to regular tensor if possible
# isregular required for type stability
isregular(::MixedTensor{1}) = true
isregular(::MixedTensor{2, dims}) where {dims} = dims[1] === dims[2]
isregular(::MixedTensor{3, dims}) where {dims} = dims[1] === dims[2] === dims[3]
isregular(::MixedTensor{4, dims}) where {dims} = dims[1] === dims[2] === dims[3] === dims[4]

function makeregular(t::MixedTensor{order,dims}) where {order,dims}
    if isregular(t)
        return Tensor{order,dims[1]}(get_data(t))
    else
        return t
    end
end
makemixed(t::Tensor{1, dim}) where {dim} = MixedTensor{1, (dim,)}(get_data(t))
makemixed(t::Tensor{2, dim}) where {dim} = MixedTensor{2, (dim, dim)}(get_data(t))
makemixed(t::Tensor{3, dim}) where {dim} = MixedTensor{3, (dim, dim, dim)}(get_data(t))
makemixed(t::Tensor{4, dim}) where {dim} = MixedTensor{4, (dim, dim, dim, dim)}(get_data(t))

#########################
# Internal constructors # Required ???
#########################
function dims_permutations(order, maxdim=3)
    # Get all permutations for the given order 
    # e.g. for order=2 we have (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)
    if order == 1
        return ntuple(i->(i,), maxdim)
    elseif order == 2
        return tuple(((i,j) for i in 1:maxdim, j in 1:maxdim)...)
    elseif order == 3
        return tuple(((i,j,k) for i in 1:maxdim, j in 1:maxdim, k in 1:maxdim)...)
    elseif order == 4
        return tuple(((i,j,k,l) for i in 1:maxdim, j in 1:maxdim, k in 1:maxdim, l in 1:maxdim)...)
    else
        throw(ArgumentError("order=$order not supported"))
    end
end

for order in (1,2,3,4)
    for dims in dims_permutations(order)
        M = n_components(MixedTensor{order, dims})
        @eval begin
            @inline MixedTensor{$order, $dims}(t::NTuple{$M, T}) where T = MixedTensor{$order, $dims, T, $M}(t)
            @inline MixedTensor{$order, $dims, T}(t::NTuple{$M}) where T = MixedTensor{$order, $dims, T, $M}(t)
        end
        if M > 1 # To avoid overwriting ::Tuple{Any}
            # Heterogeneous tuple
            @eval @inline MixedTensor{$order, $dims}(t::Tuple{Vararg{<:Any,$M}}) = MixedTensor{$order, $dims}(promote(t...))
        end
    end
end
# End internal constructors
###########################
