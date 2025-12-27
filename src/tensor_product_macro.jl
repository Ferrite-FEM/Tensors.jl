const IndexSymbols{N} = NTuple{N, Symbol}

# Given an index `name` and the `names` corresponding to the `indices`,
# return the index for `name`.
function find_index(name::Symbol, names::IndexSymbols, indices::NTuple{<:Any, Int})
    i = findfirst(Base.Fix1(===, name), names)
    i !== nothing && return indices[i]
    error("Could not find $name in si=$names")
end

# Same as above, but if not found in the first `names1`, then try to find in `names2`.
# `isdisjoint(names1, names2)` should be true when calling this function. 
function find_index(name::Symbol, names1::IndexSymbols, indices1::NTuple{<:Any, Int}, names2::IndexSymbols, indices2::NTuple{<:Any, Int})
    i = findfirst(Base.Fix1(===, name), names1)
    i !== nothing && return indices1[i]
    i = findfirst(Base.Fix1(===, name), names2)
    i !== nothing && return indices2[i]
    error("Could not find $name in names1=$names1 or names2=$names2")
end

# Return the tensor base, e.g. `Tensor{order, dim}` based on the `name` and which 
# indices belong to the tensor.
function get_tensor_type(name::Symbol, index_names::IndexSymbols, dims::NamedTuple)
    if name === :Tensor || name === :SymmetricTensor
        return getproperty(Tensors, name){length(index_names), first(dims)}
    else
        error("MixedTensor not yet supported")
    end
end

"""
    get_expression(ci::IndexSymbols, ai::IndexSymbols, bi::IndexSymbols, dims::NamedTuple; 
        TA::Symbol, TB::Symbol, TC::Symbol, use_muladd::Bool=false)

`ci` gives the output indices, ai the indices of the first tensor and bi of the second.
`dims` describe the dimension for each index name, and may be provided as `Int` if 
all are equal (i.e. not MixedTensor). 

**Examples** to get the expression for the following with `dim=2` and standard `Tensor`
inputs and outputs

* `C = A[i]*B[i]`: `get_expression((), (:i,), (:i,), 2; TA = :Tensor, TB = :Tensor)`
* `C[i] = A[i,j]*B[j]`: `get_expression((:i,), (:i, :j), (:j,), 2; TA = :Tensor, TB = :Tensor, TC = :Tensor)`
* `C[i,j] = A[i,l,m]*B[l,m,j]`: `get_expression((:i, :j), (:i, :l, :m), (:l, :m, :j), 2; TA = :Tensor, TB = :Tensor, TC = :Tensor)`

"""
function get_expression(ci::IndexSymbols, ai::IndexSymbols, bi::IndexSymbols, 
        dims::NamedTuple; TC, TA, TB, use_muladd=false
        )
    # Convert type to actual Type
    TTA = get_tensor_type(TA, ai, dims)
    TTB = get_tensor_type(TB, bi, dims)
    TTC = get_tensor_type(TC, ci, dims)

    idxA(args...) = compute_index(TTA, args...)
    idxB(args...) = compute_index(TTB, args...)

    sum_keys = tuple(sort(intersect(ai, bi))...) # The index names to sum over 

    # Validate input
    issubset(ci, union(ai, bi)) || error("All indices in ci must in either ai or bi")
    isdisjoint(sum_keys, ci) || error("Indices in ci cannot only exist once in union(ai, bi)")
    if length(ci) != (length(ai) + length(bi) - 2*length(sum_keys)) 
        error("Some indices occurs more than once in ai or bi, summation indices should occur once in ai and once in bi")
    end
    
    exps = Expr(:tuple)
    for cinds in Iterators.ProductIterator(tuple((1:dims[k] for k in ci)...))
        exa = Expr[]
        exb = Expr[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in sum_keys)...))
            ainds = tuple((find_index(a, ci, cinds, sum_keys, sinds) for a in ai)...)
            binds = tuple((find_index(b, ci, cinds, sum_keys, sinds) for b in bi)...)
            push!(exa, :(get_data(A)[$(idxA(ainds...))]))
            push!(exb, :(get_data(B)[$(idxB(binds...))]))
        end
        push!(exps.args, reducer(exa, exb, use_muladd))
    end
    return :($TTC($(remove_duplicates(TTC, exps))))
end

function get_expression(ci, ai::IndexSymbols, bi::IndexSymbols, dim::Int; kwargs...)
    dims = NamedTuple(k=>dim for k in union(ai, bi)) # Convert scalar dim to one dim for each index.
    return get_expression(ci, ai, bi, dims; kwargs...)
end

# For scalar output
function get_expression(::Tuple{}, ai::IndexSymbols, bi::IndexSymbols, 
        dims::NamedTuple; TC::Nothing=nothing, TA, TB, use_muladd=false
        )
    # Convert type to actual Type
    TTA = get_tensor_type(TA, ai, dims)
    TTB = get_tensor_type(TB, bi, dims)

    idxA(args...) = compute_index(TTA, args...)
    idxB(args...) = compute_index(TTB, args...)

    sum_keys = tuple(sort(intersect(ai, bi))...) # The index names to sum over
    if !(length(sum_keys) == length(ai) == length(bi))
        error("For scalar output, all indices in ai must be in bi, and vice versa")
    end

    exa = Expr[]
    exb = Expr[]
    for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in sum_keys)...))
        ainds = tuple((find_index(a, sum_keys, sinds) for a in ai)...)
        binds = tuple((find_index(b, sum_keys, sinds) for b in bi)...)
        push!(exa, :(get_data(A)[$(idxA(ainds...))]))
        push!(exb, :(get_data(B)[$(idxB(binds...))]))
    end
    return reducer(exa, exb, use_muladd)
end
