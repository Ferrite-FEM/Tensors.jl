#######################################################################
# Einsum engine                                                       #
#                                                                     #
# Single source of truth for all index-notation defined operations    #
# (contractions and products). An operation like                      #
#                                                                     #
#     C[i,j] = A[i,k] * B[k,j]                                        #
#                                                                     #
# is declared with `@tensorop` (see below), which expands to one      #
# `@generated` method. At generation time the planner sees the        #
# concrete argument types (order/dim/symmetry/mixed dims) and emits   #
# a flat component-tuple expression with:                             #
#   * symmetric-storage aware data indexing (compute_index)           #
#   * duplicate-product collapse with integer factors                 #
#   * muladd accumulation chains where declared                       #
#   * symmetric/mixed output-type computation, with MixedTensor       #
#     results collapsed to Tensor when all dimensions agree           #
#                                                                     #
# The emitted scalar code matches the pre-rewrite package expression- #
# for-expression (that is the performance and numerics parity target).#
#######################################################################

# One indexed argument of a term: its base type and the index names, e.g. A[i,j]
struct IndexedArg
    name::Symbol
    T::Type              # base type from get_base, e.g. Tensor{2,3}
    inds::Tuple{Vararg{Symbol}}
    elt::Type            # eltype of the concrete argument (drives SIMD lowering)
end
IndexedArg(name::Symbol, T::Type, inds::Tuple{Vararg{Symbol}}) = IndexedArg(name, T, inds, Any)

# size of a *base* type (as returned by get_base, i.e. with free eltype/M),
# for use at generation time
base_size(::Type{Tensor{order, dim}}) where {order, dim} = ntuple(_ -> dim, order)
base_size(::Type{SymmetricTensor{order, dim}}) where {order, dim} = ntuple(_ -> dim, order)
base_size(::Type{MixedTensor{order, dims}}) where {order, dims} = tuple(dims.parameters...)

# Return a NamedTuple-like list of (index name => dimension) pairs, or `nothing`
# if an index name is used with disagreeing dimensions (=> runtime DimensionMismatch).
function index_dims(args::Tuple{Vararg{IndexedArg}})
    names = Symbol[]
    dims = Int[]
    for a in args
        sz = base_size(a.T)
        length(sz) == length(a.inds) || error("index count does not match order of $(a.T)")
        for (name, d) in zip(a.inds, sz)
            i = findfirst(==(name), names)
            if i === nothing
                push!(names, name); push!(dims, d)
            elseif dims[i] != d
                return nothing
            end
        end
    end
    return names, dims
end

dim_of(names::Vector{Symbol}, dims::Vector{Int}, name::Symbol) = dims[findfirst(==(name), names)::Int]

# Compute the output type for `out_inds` given the argument types.
# Follows the pre-rewrite rules exactly:
#  * scalar for no output indices
#  * MixedTensor when output dimensions differ (collapse to Tensor handled by
#    the caller getting a regular Tensor type back when all dims agree)
#  * SymmetricTensor when every consecutive output index pair (1,2), (3,4), ...
#    provably carries symmetry from a symmetric argument
function output_type(out_inds::Tuple{Vararg{Symbol}}, args::Tuple{Vararg{IndexedArg}},
                     names::Vector{Symbol}, dims::Vector{Int})
    order = length(out_inds)
    order == 0 && return nothing # scalar
    out_dims = map(name -> dim_of(names, dims, name), out_inds)
    if !all(==(out_dims[1]), out_dims)
        return MixedTensor{order, Tuple{out_dims...}}
    end
    dim = out_dims[1]
    isodd(order) && return Tensor{order, dim}
    symmetric_output = all(2:2:order) do pair_idx
        is_symmetric_pair(args, out_inds[pair_idx - 1], out_inds[pair_idx])
    end
    return symmetric_output ? SymmetricTensor{order, dim} : Tensor{order, dim}
end

# Does some argument carry (idx1, idx2) as an adjacent (odd, even) index pair
# of a SymmetricTensor? (Note: does not detect symmetry from e.g. A[i,j]*A[j,k].)
function is_symmetric_pair(args::Tuple{Vararg{IndexedArg}}, idx1::Symbol, idx2::Symbol)
    for a in args
        nr1 = findfirst(==(idx1), a.inds)
        nr2 = findfirst(==(idx2), a.inds)
        if nr1 !== nothing && nr2 !== nothing
            if abs(nr1 - nr2) == 1 && isodd(min(nr1, nr2))
                return a.T <: SymmetricTensor
            end
        end
    end
    return false
end

# The (ia, ib) data-index products for one output component, with duplicates
# collapsed into integer factors (same algorithm and ordering as the old package).
function component_products(out_inds, oind, sum_inds, names, dims, A::IndexedArg, B::IndexedArg)
    lookup_names = (out_inds..., sum_inds...)
    prods = NTuple{2, Int}[]
    sum_ranges = ntuple(k -> 1:dim_of(names, dims, sum_inds[k]), length(sum_inds))
    for sind in Iterators.product(sum_ranges...)
        vals = (oind..., sind...)
        pos(name) = vals[findfirst(==(name), lookup_names)::Int]
        ia = compute_index(A.T, map(pos, A.inds)...)
        ib = compute_index(B.T, map(pos, B.inds)...)
        push!(prods, (ia, ib))
    end
    uniq = NTuple{2, Int}[]
    factors = Int[]
    for p in prods
        i = findfirst(==(p), uniq)
        if i === nothing
            push!(uniq, p); push!(factors, 1)
        else
            factors[i] += 1
        end
    end
    return uniq, factors
end

# Sum-of-products expression: initialized from the first product (never zero(T)),
# left-fold accumulation, muladd where requested, integer factors multiplied onto
# the first operand — exactly the old package's `reducer` emission.
function sum_expr(uniq, factors, da::Symbol, db::Symbol, madd::Bool)
    lhs(k) = begin
        f = factors[k]; ia = uniq[k][1]
        f == 1 ? :($da[$ia]) : :($f * $da[$ia])
    end
    rhs(k) = :($db[$(uniq[k][2])])
    ex = :($(lhs(1)) * $(rhs(1)))
    for k in 2:length(uniq)
        ex = madd ? :(muladd($(lhs(k)), $(rhs(k)), $ex)) :
                    :($ex + $(lhs(k)) * $(rhs(k)))
    end
    return ex
end

"""
    einsum_expr(out_inds, A::IndexedArg, B::IndexedArg; muladd = false, force_out = nothing)

Return the expression computing `Out[out_inds...] = A[...] * B[...]` with
summation over the indices shared between `A` and `B`. Used inside
`@generated` bodies (via `@tensorop`); the expression refers to the argument
variables by name and reads their data with `get_data`.

`force_out` overrides the computed output type, for callers that know the
result has more structure than the index pattern proves (e.g. symmetric
output when contracting commuting symmetric tensors).
"""
function einsum_expr(out_inds::Tuple{Vararg{Symbol}}, A::IndexedArg, B::IndexedArg; muladd::Bool = false, force_out = nothing)
    nd = index_dims((A, B))
    if nd === nothing
        return :(throw(DimensionMismatch(string("dimensions of the tensor indices do not agree when computing ",
                                                $(string(A.name, "[", join(A.inds, ","), "] * ", B.name, "[", join(B.inds, ","), "]"))))))
    end
    names, dims = nd
    sum_inds = tuple(sort(collect(intersect(A.inds, B.inds)))...)
    # validate index structure (definition-time errors, these are programmer errors)
    issubset(out_inds, union(A.inds, B.inds)) || error("output indices must appear in the term")
    isdisjoint(sum_inds, out_inds) || error("output indices cannot be summation indices")
    if length(out_inds) != (length(A.inds) + length(B.inds) - 2 * length(sum_inds))
        error("an index appears more than once in a single argument; this is not supported")
    end

    da, db = gensym(A.name), gensym(B.name)
    OutType = isempty(out_inds) ? nothing :
              (force_out === nothing ? output_type(out_inds, (A, B), names, dims) : force_out)
    comps = OutType === nothing ? [()] : base_components(OutType)
    plans = [component_products(out_inds, oind, sum_inds, names, dims, A, B) for oind in comps]

    # SIMD lowering for same-eltype hardware float arguments (the cases the old
    # hand-written simd.jl kernels covered); falls back to scalar lowering when
    # the plan lacks column structure.
    if simd_eligible(A.elt, B.elt) && !(OutType === nothing && length(sum_inds) < 2)
        # (scalar output with a single summed index — plain dot — beats the
        # tree-reduction SVec form at dim ≤ 3, so it stays on the scalar path,
        # as in the old package)
        ex = try_simd_expr(OutType, plans, da, db, :(get_data($(A.name))), :(get_data($(B.name))), A.elt)
        ex !== nothing && return ex
    end

    exprs = [sum_expr(uniq, factors, da, db, muladd) for (uniq, factors) in plans]
    if OutType === nothing
        return quote
            $da = get_data($(A.name)); $db = get_data($(B.name))
            @inbounds return $(exprs[1])
        end
    end
    return quote
        $da = get_data($(A.name)); $db = get_data($(B.name))
        @inbounds return $(OutType)($(Expr(:tuple, exprs...)))
    end
end

#############
# @tensorop #
#############
"""
    @tensorop function op(A::TensorType, B::TensorType) where {...}
        C[i,j] = A[i,k] * B[k,j]
    end

Define a tensor operation from index notation. Expands to a `@generated`
method with the given signature whose body is produced by [`einsum_expr`](@ref)
from the concrete argument types. The left-hand side gives the output indices
(`C[] = ...` or `C = ...` for scalar output); wrap the assignment in
`@muladd ...` to accumulate with `muladd`.

The generated method covers `Tensor`, `SymmetricTensor` and `MixedTensor`
arguments of any dimension in one definition; the output type is computed
from the index structure (symmetric output for symmetric-carrying index
pairs, `MixedTensor` collapsed to `Tensor` when possible).
"""
macro tensorop(fdef::Expr)
    fdef.head === :function || error("@tensorop expects a function definition")
    sig = fdef.args[1]
    body = fdef.args[2]
    # unwrap where-clauses to find the call
    call = sig
    while call isa Expr && call.head === :where
        call = call.args[1]
    end
    call isa Expr && call.head === :call || error("@tensorop expects a function definition")
    argnames = Symbol[]
    for arg in call.args[2:end]
        arg isa Expr && arg.head === :(::) && length(arg.args) == 2 ||
            error("@tensorop arguments must be typed, e.g. A::SecondOrderTensor")
        push!(argnames, arg.args[1])
    end
    # extract the single index-assignment statement from the body
    stmts = [a for a in body.args if !(a isa LineNumberNode)]
    length(stmts) == 1 || error("@tensorop body must be a single index assignment")
    stmt = stmts[1]
    use_muladd = false
    if stmt isa Expr && stmt.head === :macrocall && stmt.args[1] === Symbol("@muladd")
        use_muladd = true
        stmt = [a for a in stmt.args[2:end] if !(a isa LineNumberNode)][]
    end
    stmt isa Expr && stmt.head === :(=) || error("@tensorop body must be an assignment, e.g. C[i,j] = A[i,k] * B[k,j]")
    lhs, rhs = stmt.args[1], stmt.args[2]
    out_inds = if lhs isa Symbol
        () # scalar output
    elseif lhs isa Expr && lhs.head === :ref
        tuple(lhs.args[2:end]...)
    else
        error("@tensorop left-hand side must be `C` (scalar) or `C[i,j,...]`")
    end
    # parse rhs: product of indexed arguments, e.g. A[i,k] * B[k,j]
    rhs isa Expr && rhs.head === :call && rhs.args[1] === :* ||
        error("@tensorop right-hand side must be a product of indexed tensors")
    refs = rhs.args[2:end]
    length(refs) == 2 || error("@tensorop currently supports exactly two arguments")
    argspecs = map(refs) do r
        r isa Expr && r.head === :ref || error("expected an indexed tensor like A[i,j], got $r")
        name = r.args[1]
        name in argnames || error("$(name) is not an argument of the function")
        (name, tuple(r.args[2:end]...))
    end
    ai_name, ai_inds = argspecs[1]
    bi_name, bi_inds = argspecs[2]
    genbody = quote
        expr = einsum_expr($(QuoteNode(out_inds)),
                           IndexedArg($(QuoteNode(ai_name)), get_base($ai_name), $(QuoteNode(ai_inds)), eltype($ai_name)),
                           IndexedArg($(QuoteNode(bi_name)), get_base($bi_name), $(QuoteNode(bi_inds)), eltype($bi_name));
                           muladd = $use_muladd)
        return quote
            $(Expr(:meta, :inline))
            $expr
        end
    end
    gen = Expr(:function, sig, genbody)
    return esc(Expr(:macrocall, Symbol("@generated"), __source__, gen))
end
