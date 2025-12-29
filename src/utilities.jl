function tensor_create_linear(T::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}, Type{MixedTensor{order, dim}}}, f) where {order, dim}
    return Expr(:tuple, [f(i) for i=1:n_components(T)]...)
end

function tensor_create(::Type{Tensor{order, dim}}, f) where {order, dim}
    return tensor_create(MixedTensor{order, ntuple(_ -> dim, order)}, f)
end

function tensor_create(::Type{MixedTensor{order, dims}}, f) where {order, dims}
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dims[1]]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dims[1], j=1:dims[2]]...)
    elseif order == 3
        ex = Expr(:tuple, [f(i,j,k) for i=1:dims[1], j=1:dims[2], k=1:dims[3]]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dims[1], j=1:dims[2], k = 1:dims[3], l = 1:dims[4]]...)
    end
    return ex
end

function tensor_create(::Type{SymmetricTensor{order, dim}}, f) where {order, dim}
    ex = Expr(:tuple)
    if order == 2
        for j in 1:dim, i in j:dim
            push!(ex.args, f(i, j))
        end
    elseif order == 4
        for l in 1:dim, k in l:dim, j in 1:dim, i in j:dim
            push!(ex.args, f(i, j, k, l))
        end
    end
    return ex
end


# reduced two expressions by summing the products
# madd = true uses muladd instructions which is faster
# in some cases, like in double contraction
function reducer(ex1i, ex2i, madd=false)
    ex1, ex2 = remove_duplicates(ex1i, ex2i)
    N = length(ex1)
    expr = :($(ex1[1]) * $(ex2[1]))

    for i in 2:N
        expr = madd ? :(muladd($(ex1[i]), $(ex2[i]), $expr)) :
                      :($(expr) + $(ex1[i]) * $(ex2[i]))
    end
    return expr
end

function remove_duplicates(ex1in, ex2in)
    ex1out, ex2out = Expr[], Expr[]
    exout = Expr[]
    factors = ones(Int, length(ex1in))

    for (ex1ine, ex2ine) in zip(ex1in, ex2in)
        prod = :($ex1ine * $ex2ine)
        i = findfirst(isequal(prod), exout) # check if this product exist in the output
        if i == nothing # this product does not exist yet
            push!(ex1out, ex1ine)
            push!(ex2out, ex2ine)
            push!(exout, prod)
        else # found a duplicate
            factors[i] += 1
        end
    end
    for i in 1:length(ex1out)
        factors[i] != 1 && (ex1out[i] = :($(factors[i]) * $(ex1out[i])))
    end
    return ex1out, ex2out
end

# check symmetry and return
remove_duplicates(::Type{<:Tensor}, ex) = ex # do nothing if return type is a Tensor
remove_duplicates(::Type{<:MixedTensor}, ex) = ex
function remove_duplicates(::Type{SymmetricTensor{order, dim}}, ex) where {order, dim}
    ex.args = ex.args[SYMMETRIC_INDICES[order][dim]]
    return ex
end

# unit stripping if necessary
function ustrip(S::SymmetricTensor{order,dim,T}) where {order, dim, T}
    ou = oneunit(T)
    if typeof(ou / ou) === T # no units
        return S
    else # units, so strip them by dividing with oneunit(T)
        return SymmetricTensor{order,dim}(map(x -> x / ou, S.data))
    end
end

struct IndexedTensor{TB, order, NT}
    inds::NTuple{order, Symbol}   # index nr => index name (e.g. `(:i, :j)`)
    typename::Symbol            # `:Tensor`, `:SymmetricTensor`, or `:MixedTensor`
    dims::NT                    # Dimension for each index name
    name::Symbol                # Variable name, e.g. `:A`
    function IndexedTensor{TT}(inds::NTuple{order, Symbol}, name::Symbol
        ) where {order, dim, TT <: Union{Tensor{order, dim}, SymmetricTensor{order, dim}, MixedTensor{order, dim}}}
        dimnrs = (isa(dim, Int) ? ntuple(_ -> dim, order) : dim)::NTuple{order, Int}
        dims = NamedTuple{inds}(dimnrs)
        TB = get_base(TT)
        return new{TB, order, typeof(dims)}(inds, nameof(TB), dims, name)
    end
end
get_base(::IndexedTensor{TB}) where {TB} = TB

const IndexedTensorTerm{N} = Tuple{Vararg{IndexedTensor, N}}

# Given an index `name` and the `names` corresponding to the `indices`,
# return the index for `name`.
function find_index(name::Symbol, names::NTuple{<:Any, Symbol}, indices::NTuple{<:Any, Int})
    i = findfirst(Base.Fix1(===, name), names)
    i !== nothing && return indices[i]
    error("Could not find $name in si=$names")
end

# Same as above, but if not found in the first `names1`, then try to find in `names2`.
# `isdisjoint(names1, names2)` should be true when calling this function. 
function find_index(name::Symbol, names1::NTuple{<:Any, Symbol}, indices1::NTuple{<:Any, Int}, names2::NTuple{<:Any, Symbol}, indices2::NTuple{<:Any, Int})
    i = findfirst(Base.Fix1(===, name), names1)
    i !== nothing && return indices1[i]
    i = findfirst(Base.Fix1(===, name), names2)
    i !== nothing && return indices2[i]
    error("Could not find $name in names1=$names1 or names2=$names2")
end

"""
    get_term_expression(out_inds::NTuple{<:Any, Symbol}, term::IndexedTensorTerm; use_muladd::Bool = false)

`out_inds` gives the output indices, and term gives the indexed tensor expression consisting of `IndexedTensor`

**Examples** to get the expression for the following standard `Tensor` inputs

* `C = A[i]*B[i]`: `get_term_expression((), (IndexedTensor{Tensor{1,2}}((:i,), :A), IndexedTensor{Tensor{1,2}}((:i,), :B)))`
* `C[i] = A[i,j]*B[j]`: `get_term_expression((:i,), (IndexedTensor{Tensor{2,2}}((:i, :j), :A), IndexedTensor{Tensor{1,2}}((:j,), :B)))`
* `C[i,j] = A[i,l,m]*B[l,m,j]`: `get_term_expression((:i, :j), (IndexedTensor{Tensor{3,2}}((:i, :l, :m), :A), IndexedTensor{Tensor{3,2}}((:l, :m, :j), :B)))`

"""
function get_term_expression end

function get_term_expression(out_inds::NTuple{<:Any, Symbol}, term::IndexedTensorTerm{2}; use_muladd = false)
    # Return the expression for the tuple to fill the output tensor with, not considering that 
    # the output tensor might be symmetric (this should be done on the complete sum of terms if applicable)

    A, B = term
    TA, TB = get_base.((A, B))
    ai = A.inds
    bi = B.inds
    dims = get_term_dims(term)
    
    idxA(args...) = compute_index(TA, args...)
    idxB(args...) = compute_index(TB, args...)

    sum_inds = tuple(sort(intersect(ai, bi))...) # The index names to sum over 

    # Validate input
    issubset(out_inds, union(ai, bi)) || error("All indices in `out_inds` must be in term")
    isdisjoint(sum_inds, out_inds) || error("Indices in `out_inds` can only exist once in the term")
    if length(out_inds) != (length(ai) + length(bi) - 2*length(sum_inds)) 
        error("Some indices occurs more than once in an `IndexedTensor` in the term, this is currently not supported")
    end
    
    expr = Expr(:tuple)
    for o in Iterators.ProductIterator(tuple((1:dims[k] for k in out_inds)...))
        exa = Expr[]
        exb = Expr[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in sum_inds)...))
            ainds = tuple((find_index(a, out_inds, o, sum_inds, sinds) for a in ai)...)
            binds = tuple((find_index(b, out_inds, o, sum_inds, sinds) for b in bi)...)
            push!(exa, :(get_data($(A.name))[$(idxA(ainds...))]))
            push!(exb, :(get_data($(B.name))[$(idxB(binds...))]))
        end
        push!(expr.args, reducer(exa, exb, use_muladd))
    end
    return expr
end

function get_term_expression(::Tuple{}, term::IndexedTensorTerm{2}; use_muladd = false)
    A, B = term
    TA, TB = get_base.((A, B))
    ai = A.inds
    bi = B.inds
    dims = get_term_dims(term)
    
    idxA(args...) = compute_index(TA, args...)
    idxB(args...) = compute_index(TB, args...)

    sum_inds = tuple(sort(intersect(ai, bi))...) # The index names to sum over 
    if !(length(sum_inds) == length(ai) == length(bi))
        error("For scalar output, all indices in ai must be in bi, and vice versa")
    end

    exa = Expr[]
    exb = Expr[]
    for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in sum_inds)...))
        ainds = tuple((find_index(a, sum_inds, sinds) for a in ai)...)
        binds = tuple((find_index(b, sum_inds, sinds) for b in bi)...)
        push!(exa, :(get_data(A)[$(idxA(ainds...))]))
        push!(exb, :(get_data(B)[$(idxB(binds...))]))
    end
    return reducer(exa, exb, use_muladd)
end

function get_term_dims(term::IndexedTensorTerm)
    dims = Dict{Symbol, Int}()
    for it in term
        for (iname, dim) in zip(keys(it.dims), it.dims)
            # TODO: This should be checked at the top level and result in a MethodError
            if haskey(dims, iname) && dims[iname] != dim
                error("index $iname as different dims ($dim and $(dims[iname]))\nThis means that you've called the function with incompatible tensor dimensions")
            end
            dims[iname] = dim
        end
    end
    return dims
end

"""
    get_expression(out_inds::NTuple{<:Any, Symbol}, rhs::Expr, tensor_types::NamedTuple; kwargs...)

TODO: Write docstring and add limitations. Note that this is the implementer-facing function. 
"""
function get_expression(out_inds::NTuple{<:Any, Symbol}, rhs::Expr, tensor_types::NamedTuple; kwargs...)
    rhs.head === :call || error("The right-hand-side must be a function call")
    (rhs.args[1] == :+ || rhs.args[1] == :-) && error("Multiple terms currently not supported")
    rhs.args[1] == :* || error("Only multiplication between tensors supported")
    term = get_term(rhs, tensor_types)
    expr = get_term_expression(out_inds, term; kwargs...)
    if length(out_inds) == 0 # Scalar
        return expr
    else
        TO = get_output_type(out_inds, term)
        expr_red = remove_duplicates(TO, expr)
        return :($(TO)($expr_red))
    end
end

function get_term(rhs::Expr, tensor_types::NamedTuple)
    # Already validated that rhs.head == :call and rhs.args[1] == :*
    return ntuple(length(rhs.args) - 1) do i
        it_expr = rhs.args[i + 1]
        it_expr.head == :ref || error("expected an indexed tensor expression, e.g. `A[i,j]`, but got: `$(it_expr)`")
        name = it_expr.args[1]
        inds = tuple(it_expr.args[2:end]...)
        isa(inds, NTuple{<:Any, Symbol}) || error("malformatted indexed tensor expression = `$(it_expr)`")
        TT = tensor_types[name]
        IndexedTensor{TT}(inds, name)
    end
end

function get_output_type(out_inds::NTuple{<:Any, Symbol}, term::IndexedTensorTerm{2})
    dims = get_term_dims(term)
    order = length(out_inds)
    order == 0 && return :scalar
    out_dims = map(k -> dims[k], out_inds)
    if all(d -> d == out_dims[1], out_dims) # allequal
        # Not MixedTensor, need to see if SymmetricTensor
        dim = dims[first(out_inds)]
        isodd(length(out_inds)) && return Tensor{order, dim}
        symmetric_output = all(2:2:order) do pair_idx
            is_symmetric_indices(term, out_inds[pair_idx - 1], out_inds[pair_idx])
        end
        return symmetric_output ? SymmetricTensor{order, dim} : Tensor{order, dim}    
    else
        return MixedTensor{order, out_dims}
    end
end

function is_symmetric_indices(term::IndexedTensorTerm, idx1::Symbol, idx2::Symbol)
    # Note: Currently doesn't add symmetry if we have e.g. A[i,j] * A[j,k] for symmetric A
    for it in term
        nr1 = findfirst(k -> k == idx1, it.inds)
        nr2 = findfirst(k -> k == idx2, it.inds)
        if nr1 !== nothing && nr2 !== nothing
            if abs(nr1 - nr2) == 1 && isodd(min(nr1, nr2))
                return it.typename === :SymmetricTensor
            end
        end
    end
    return false
end
