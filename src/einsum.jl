# ========================================================================== #
#                                                                            #
#                             The einsum engine                              #
#                                                                            #
# ========================================================================== #
#
# Every product and contraction in this package is *declared* in index
# notation and *compiled* by the code in this file. A declaration looks like
#
#     @tensorop function dot(A::SecondOrderTensor, B::SecondOrderTensor)
#         C[i, j] = A[i, k] * B[k, j]
#     end
#
# and turns into one `@generated` method. When Julia compiles that method for
# concrete argument types, `einsum_expr` below builds the method body in four
# steps:
#
#     1. bookkeeping — which index names exist, what dimension each one has,
#        which are summed over (they appear in two arguments) and which are
#        output indices (they appear once, on the left-hand side);
#     2. output type — Tensor, SymmetricTensor or MixedTensor, computed from
#        the index structure;
#     3. the plan — for every stored component of the output, the list of
#        scalar products that sum to it, as plain integer indices into the
#        arguments' data tuples;
#     4. lowering — turn the plan into an expression: either straight-line
#        scalar arithmetic (below) or SIMD vector code (simd_lowering.jl).
#
# A worked example, `dot` as declared above with a symmetric and a regular
# argument in two dimensions, `A::SymmetricTensor{2,2}` and `B::Tensor{2,2}`:
#
#     A is stored packed as   (a11, a21, a22)            (3 entries)
#     B is stored column-major (b11, b21, b12, b22)      (4 entries)
#
# The output C[i,j] = A[i,k]*B[k,j] is a full Tensor{2,2} with 4 components.
# The plan maps each component to (data index into A, data index into B)
# pairs — note how A[i, k] reads packed entries via the symmetric layout
# (A[1,2] and A[2,1] are both data entry 2):
#
#     C[1,1] = A[1,1]*B[1,1] + A[1,2]*B[2,1]   plan: [(1,1), (2,2)]
#     C[2,1] = A[2,1]*B[1,1] + A[2,2]*B[2,1]   plan: [(2,1), (3,2)]
#     C[1,2] = A[1,1]*B[1,2] + A[1,2]*B[2,2]   plan: [(1,3), (2,4)]
#     C[2,2] = A[2,1]*B[1,2] + A[2,2]*B[2,2]   plan: [(2,3), (3,4)]
#
# and the emitted method body is (with `@muladd` in the declaration):
#
#     $(Expr(:meta, :inline))
#     _d1 = Tensors.get_data(A)
#     _d2 = Tensors.get_data(B)
#     @inbounds return Tensor{2, 2}((
#         muladd(_d1[2], _d2[2], _d1[1] * _d2[1]),
#         muladd(_d1[3], _d2[2], _d1[2] * _d2[1]),
#         muladd(_d1[2], _d2[4], _d1[1] * _d2[3]),
#         muladd(_d1[3], _d2[4], _d1[2] * _d2[3])))
#
# When both sides of a product are the same packed entry the plan collapses
# them into an integer multiplicity instead of computing them twice. The
# scalar `dcontract(A, B) = A[i,j]*B[i,j]` of two SymmetricTensor{2,2} sums
# over the full 2×2 index grid, but only 3 distinct products exist:
#
#     plan: [(1,1), (2,2), (3,3)] with multiplicities [1, 2, 1]
#     emitted:  _d1[1] * _d2[1] + (2 * _d1[2]) * _d2[2] + _d1[3] * _d2[3]
#
# (A multiplicity is a product of factors of 2, one per symmetric index pair
# whose two orderings hit the same packed entry — so it is a power of two,
# 1, 2 or 4, and the scaling is exact for floats.)
#
# The scalar lowering in this file is the semantic reference: it reproduces
# the pre-rewrite package expression-for-expression. The SIMD lowering
# evaluates the same products in the same order per output component, but as
# muladd chains regardless of the `@muladd` flag and (for scalar output) a
# horizontal vector sum — exactly what the old hand-written kernels did, so
# a given operation and eltype computes what it always computed. See the
# contract note in simd_lowering.jl.
#
# Vocabulary used throughout (defined elsewhere in the package):
#   get_data(A)            the NTuple backing a tensor
#   get_base(T)            the type with eltype/length stripped, e.g. Tensor{2,3}
#   base_size(T)           dimensions per index, e.g. (3, 3)
#   compute_index(T, i...) Cartesian index -> position in the data tuple;
#                          encodes column-major and symmetric-packed layouts
#                          (indexing.jl)
#   base_components(T)     the Cartesian indices of the stored components, in
#                          data-tuple (column-major) order (indexing.jl)

# -------------------------------------------------------------------------- #
# An argument as it appears in a declaration: `A[i,k]` becomes
# IndexedArg(:A, Tensor{2,2}, (:i,:k), Float64) once the argument type is
# known. `T` is the base type (`get_base`, i.e. eltype- and M-free), `elt`
# the concrete element type — the SIMD lowering only applies to hardware
# floats and needs to see it.
struct IndexedArg
    name::Symbol
    T::Type
    inds::Tuple{Vararg{Symbol}}
    elt::Type
end
IndexedArg(name::Symbol, T::Type, inds::Tuple{Vararg{Symbol}}) = IndexedArg(name, T, inds, Any)

# ------------------------------ step 1 ------------------------------------ #
# Index bookkeeping.

# Size of a *base* type (with free eltype/M), usable at code-generation time.
base_size(::Type{Tensor{order, dim}}) where {order, dim} = ntuple(_ -> dim, order)
base_size(::Type{SymmetricTensor{order, dim}}) where {order, dim} = ntuple(_ -> dim, order)
base_size(::Type{MixedTensor{order, dims}}) where {order, dims} = tuple(dims.parameters...)

# Collect every index name and its dimension, in first-appearance order.
# Returns `nothing` when an index name is used with two different dimensions
# (possible with MixedTensor arguments) — the caller then emits code that
# throws a runtime `DimensionMismatch`.
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

# ------------------------------ step 2 ------------------------------------ #
# Output type from the index structure. The rules (identical to the
# pre-rewrite package):
#
#   * no output indices                      -> scalar
#   * output dimensions differ               -> MixedTensor{order, Tuple{dims...}}
#     (equal dimensions never produce a MixedTensor, so mixed results
#     "collapse" to regular tensors by construction)
#   * every consecutive output index pair (1,2), (3,4), ... provably
#     symmetric                              -> SymmetricTensor
#   * otherwise                              -> Tensor
#
# Examples: in `otimes(A,B): C[i,j,k,l] = A[i,j]*B[k,l]` with both arguments
# symmetric, the pair (i,j) sits adjacent in symmetric A and (k,l) in
# symmetric B, so the result is a SymmetricTensor{4}. In
# `dot(A,B): C[i,j] = A[i,k]*B[k,j]` no single argument carries (i,j), so the
# result is a full Tensor{2} even for symmetric arguments.
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

# Does some argument carry (idx1, idx2) as one of its own symmetric index
# pairs? Symmetric storage pairs indices as (1,2) and (3,4), so the two names
# must sit adjacent at an odd/even position of a SymmetricTensor argument.
# (This does not detect "accidental" symmetry, e.g. A[i,k]*A[k,j] for
# symmetric A — same as the pre-rewrite package.)
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

# ------------------------------ step 3 ------------------------------------ #
# The plan for one output component: which scalar products sum to it.
#
# Given the output component's Cartesian index `oind` (e.g. `(2, 1)` for
# C[2,1]), loop the summation indices over their ranges and record, for each
# point of that grid, where every argument reads its data — as a tuple of
# plain linear indices into the arguments' data tuples (`compute_index` maps
# Cartesian to storage index and knows about symmetric packing).
#
# Products that turn out identical (both reads hit the same packed entries,
# which happens when summing over symmetric storage) are recorded once with
# an integer multiplicity.
#
# Returns `(products, mults)`, e.g. for C[1,1] in the header's `dot` example
# `([(1, 1), (2, 2)], [1, 1])`: two products, each read once.
function component_products(out_inds, oind, sum_inds, names, dims, args::IndexedArg...)
    N = length(args)
    # value of each index name at one point of the summation grid:
    # output indices are fixed to `oind`, summation indices take `sind`
    lookup_names = (out_inds..., sum_inds...)
    products = NTuple{N, Int}[]
    sum_ranges = ntuple(k -> 1:dim_of(names, dims, sum_inds[k]), length(sum_inds))
    for sind in Iterators.product(sum_ranges...) # first summation index fastest
        vals = (oind..., sind...)
        pos(name) = vals[findfirst(==(name), lookup_names)::Int]
        push!(products, ntuple(n -> compute_index(args[n].T, map(pos, args[n].inds)...), N))
    end
    # collapse duplicates into multiplicities, keeping first-appearance order
    unique_products = NTuple{N, Int}[]
    mults = Int[]
    for p in products
        i = findfirst(==(p), unique_products)
        if i === nothing
            push!(unique_products, p); push!(mults, 1)
        else
            mults[i] += 1
        end
    end
    return unique_products, mults
end

# ----------------------------- step 4a ------------------------------------ #
# Scalar lowering of one component plan: a left-to-right accumulation over
# the products,
#
#     acc = p₁;  acc = acc + p₂;  ...          or, with muladd:
#     acc = p₁;  acc = muladd(aₖ, bₖ, acc); ...
#
# The chain starts from the first product (never from `zero(T)`), and a
# multiplicity scales the first factor: `(2 * _d1[2]) * _d2[2]`. For more
# than two arguments the first N-1 data reads form the muladd operand and
# the last one the multiplier: `muladd(_d1[k] * _d2[ikjl], _d3[l], acc)`.
# All of this matches the pre-rewrite emission exactly.
function sum_expr(products, mults, ds::NTuple{N, Symbol}, madd::Bool) where {N}
    ref(k, n) = :($(ds[n])[$(products[k][n])])           # k-th product, n-th argument
    function first_factors(k)                            # product of reads 1..N-1
        m = mults[k]
        # (a local named `ex` would alias the accumulator below — inner
        # functions share the locals of their enclosing scope)
        e = m == 1 ? ref(k, 1) : :($m * $(ref(k, 1)))
        for n in 2:(N - 1)
            e = :($e * $(ref(k, n)))
        end
        return e
    end
    last_factor(k) = ref(k, N)
    ex = :($(first_factors(1)) * $(last_factor(1)))
    for k in 2:length(products)
        ex = madd ? :(muladd($(first_factors(k)), $(last_factor(k)), $ex)) :
                    :($ex + $(first_factors(k)) * $(last_factor(k)))
    end
    return ex
end

# ---------------------------- the driver ----------------------------------- #
"""
    einsum_expr(out_inds, args::IndexedArg...; muladd = false, force_out = nothing)

Build the method body computing `Out[out_inds...] = args[1][...] * args[2][...] * ...`,
summing over each index that appears in two of the arguments. This is the
function `@tensorop`-generated methods call at code-generation time; see the
header of this file for the pipeline and a worked example.

`force_out` overrides the computed output type, for callers that know the
result has more structure than the index pattern proves (e.g. `_powdot`,
where powers of one symmetric tensor commute and stay symmetric).
"""
function einsum_expr(out_inds::Tuple{Vararg{Symbol}}, args::IndexedArg...; muladd::Bool = false, force_out = nothing)
    N = length(args)
    N >= 2 || error("einsum_expr needs at least two arguments")

    # -- step 1: index bookkeeping ------------------------------------------
    nd = index_dims(args)
    if nd === nothing
        # disagreeing dimensions for an index name: a *runtime* error — this
        # method body is being generated for, e.g., dot of two MixedTensors
        # with incompatible axes
        term = join((string(a.name, "[", join(a.inds, ","), "]") for a in args), " * ")
        return :(throw(DimensionMismatch(string("dimensions of the tensor indices do not agree when computing ", $term))))
    end
    names, dims = nd
    # every index must appear either once (an output index) or in exactly two
    # arguments (a summation index) — violations are *definition-time* errors,
    # i.e. mistakes in an operation declaration
    for a in args
        allunique(a.inds) || error("an index appears more than once in a single argument; this is not supported")
    end
    counts = Dict(name => count(a -> name in a.inds, args) for name in names)
    any(>(2), values(counts)) && error("an index cannot appear in more than two arguments")
    # sorted alphabetically: this fixes the summation (and hence rounding)
    # order deterministically, and matches the pre-rewrite package
    sum_inds = tuple(sort(filter(name -> counts[name] == 2, names))...)
    issubset(out_inds, names) || error("output indices must appear in the term")
    isdisjoint(sum_inds, out_inds) || error("output indices cannot be summation indices")
    for name in names
        counts[name] == 1 && name ∉ out_inds && error("index $name appears only once and is not an output index")
    end

    # -- step 2: output type ------------------------------------------------
    OutType = isempty(out_inds) ? nothing :
              (force_out === nothing ? output_type(out_inds, args, names, dims) : force_out)

    # -- step 3: one plan per stored output component -----------------------
    # `plans :: Vector{Tuple{Vector{NTuple{N,Int}}, Vector{Int}}}` — one
    # (products, mults) entry per stored output component, in base_components
    # (i.e. data-tuple) order. The SIMD lowering relies on that ordering.
    comps = OutType === nothing ? [()] : base_components(OutType)
    plans = [component_products(out_inds, oind, sum_inds, names, dims, args...) for oind in comps]

    # -- step 4: lowering ---------------------------------------------------
    # `_d1`, `_d2`, ... hold the arguments' data tuples in the emitted code
    ds = ntuple(n -> Symbol(:_d, n), N)
    @assert all(a -> a.name ∉ ds, args)
    inlinemeta = Expr(:meta, :inline)

    # SIMD lowering, for binary operations on same-eltype hardware floats
    # (the cases the old hand-written simd.jl kernels covered); falls back to
    # the scalar lowering when the plan has no column structure. Exception:
    # a plain vector dot (scalar output, one summed index) always stays
    # scalar — the SVec-and-horizontal-sum form loses at the dimensions this
    # package supports (1, 2, 3), and the old package kept it scalar too.
    plain_dot = OutType === nothing && length(sum_inds) < 2
    if N == 2 && simd_eligible(args[1].elt, args[2].elt) && !plain_dot
        r = try_simd_expr(OutType, plans, ds[1], ds[2],
                          :(Tensors.get_data($(args[1].name))), :(Tensors.get_data($(args[2].name))), args[1].elt)
        if r !== nothing
            ex, inline = r
            return inline ? Expr(:block, inlinemeta, ex) : ex
        end
    end

    # scalar lowering
    databind = [:($(ds[n]) = Tensors.get_data($(args[n].name))) for n in 1:N]
    exprs = [sum_expr(products, mults, ds, muladd) for (products, mults) in plans]
    if OutType === nothing
        return quote
            $inlinemeta
            $(databind...)
            @inbounds return $(exprs[1])
        end
    end
    return quote
        $inlinemeta
        $(databind...)
        @inbounds return $(OutType)($(Expr(:tuple, exprs...)))
    end
end

# ---------------------------- the macro ------------------------------------ #
"""
    @tensorop function op(A::TensorType, B::TensorType) where {...}
        C[i,j] = A[i,k] * B[k,j]
    end

Define a tensor operation from index notation. The left-hand side gives the
output indices (a bare `C = ...` declares scalar output); an index appearing
in two arguments is summed over; wrapping the assignment in `@muladd ...`
makes the summation accumulate with `muladd`. Products of more than two
tensors are allowed (e.g. `dotdot`).

The declaration expands to a single `@generated` method with the given
signature, whose body is built by [`einsum_expr`](@ref) once the argument
types are known. One declaration therefore covers `Tensor`, `SymmetricTensor`
and `MixedTensor` arguments of every dimension and element type, with the
output type computed from the index structure. The example above expands to:

```julia
@generated function op(A::TensorType, B::TensorType) where {...}
    return Tensors.einsum_expr((:i, :j),
        Tensors.IndexedArg(:A, Tensors.get_base(A), (:i, :k), eltype(A)),
        Tensors.IndexedArg(:B, Tensors.get_base(B), (:k, :j), eltype(B));
        muladd = false)
end
```
"""
macro tensorop(fdef::Expr)
    # -- pick apart `function op(A::..., B::...) where {...}; <stmt>; end` ---
    fdef.head === :function || error("@tensorop expects a function definition")
    sig = fdef.args[1]
    body = fdef.args[2]
    call = sig
    while call isa Expr && call.head === :where # unwrap where-clauses
        call = call.args[1]
    end
    call isa Expr && call.head === :call || error("@tensorop expects a function definition")
    argnames = Symbol[]
    for arg in call.args[2:end]
        arg isa Expr && arg.head === :(::) && length(arg.args) == 2 ||
            error("@tensorop arguments must be typed, e.g. A::SecondOrderTensor")
        push!(argnames, arg.args[1])
    end

    # -- the body must be a single (possibly @muladd-wrapped) assignment -----
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

    # -- left-hand side: the output indices ----------------------------------
    out_inds = if lhs isa Symbol
        () # scalar output
    elseif lhs isa Expr && lhs.head === :ref
        tuple(lhs.args[2:end]...)
    else
        error("@tensorop left-hand side must be `C` (scalar) or `C[i,j,...]`")
    end

    # -- right-hand side: a product of indexed arguments ---------------------
    rhs isa Expr && rhs.head === :call && rhs.args[1] === :* ||
        error("@tensorop right-hand side must be a product of indexed tensors")
    refs = rhs.args[2:end]
    length(refs) >= 2 || error("@tensorop needs at least two arguments in the product")
    argspecs = map(refs) do r
        r isa Expr && r.head === :ref || error("expected an indexed tensor like A[i,j], got $r")
        name = r.args[1]
        name in argnames || error("$(name) is not an argument of the function")
        (name, tuple(r.args[2:end]...))
    end

    # -- assemble the @generated method (see the docstring for its shape) ----
    iargs = [:(Tensors.IndexedArg($(QuoteNode(name)), Tensors.get_base($name), $(QuoteNode(inds)), eltype($name)))
             for (name, inds) in argspecs]
    genbody = quote
        return Tensors.einsum_expr($(QuoteNode(out_inds)), $(iargs...); muladd = $use_muladd)
    end
    gen = Expr(:function, sig, genbody)
    return esc(Expr(:macrocall, Symbol("@generated"), __source__, gen))
end
