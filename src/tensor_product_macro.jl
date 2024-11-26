# This file defines the following macros 
# * @tensor_product: Produce efficient unrolled code from index expression
# * @foreach: macro-loop
# 
# The main "workhorse" is the `get_expression` function
# 
# First, `get_expression` gives the logic for generating the expression.
# Then, the `tensor_product` macro interprets a convenient input format 
# to return a function with the expression from `get_expression`. 
# 
# ======================================================================================
# `get_expression`
# ======================================================================================
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

# ======================================================================================
# `@tensor_product`
# ======================================================================================

# Information extracted about each argument in the function header
struct ArgInfo
    name::Symbol
    type::Symbol
    order::Int
    dim::Union{Int,Tuple{<:Any,Int}}
end
ArgInfo(;name, type, order, dim) = ArgInfo(name, type, order, dim)

# Information extracted about term in the index expression in the function body
struct TermInfo{N}
    name::Symbol
    inds::IndexSymbols{N}
end

TermInfo(term::Symbol) = TermInfo(term, ())
function TermInfo(term::Expr)
    term.head === :ref || error("TermInfo requires expression with head=:ref, but head = ", term.head)
    name = term.args[1]
    inds = tuple(term.args[2:end]...)
    return TermInfo(name, inds)
end

function extract_arginfo(arg_expr::Expr)
    if arg_expr.head !== :(::)
        error("Expected type specification arg_expr, but got expr with head $(arg_expr.head)")
    end
    @assert length(arg_expr.args) == 2
    name = arg_expr.args[1]
    @assert name isa Symbol
    curly = arg_expr.args[2]
    @assert curly.head === :curly
    type = curly.args[1]
    if type ∉ (:Tensor, :SymmetricTensor, #=MixedTensor=#)
        error("type = $type was unexpected")
    end
    order = curly.args[2]::Int  # Type-assert required, if user pass e.g. Tensor{2,dim},
    dim = curly.args[3]::Int    # the macro will not work, need to pass explicit numbers in the Expr.
    isa(dim, Int) || error("dim = $dim was unexpected, curlyargs = $(curly.args)")
    return ArgInfo(;name, type, order, dim)
end

function extract_header_arginfos(header::Expr)
    header.head === :call || error("header expression with head=$(header.head) is not supported")
    Ainfo = extract_arginfo(header.args[2])
    Binfo = extract_arginfo(header.args[3])
    return Ainfo, Binfo
end

function extract_body_information(body::Expr)
    body.head === :block || error("Expecting a block type expression")
    @assert all(isa(a, LineNumberNode) for a in body.args[1:(length(body.args)-1)])
    @assert body.args[end].head === :(=) # Should be an assignment
    expr = body.args[end].args
    Cinfo = TermInfo(expr[1])
    @assert expr[2].head === :call
    @assert expr[2].args[1] === :*
    Ainfo = TermInfo(expr[2].args[2])
    Binfo = TermInfo(expr[2].args[3])
    return Cinfo, Ainfo, Binfo
end

function check_arg_expr_consistency(head, body)
    head.name === body.name || error("head ($head) and body ($body) variable names don't match")
    if length(body.inds) !== head.order
        ninds = length(body.inds)
        error("head for $(head.name) specifices tensor of order $(head.order), but index expression has only $ninds ($(body.inds))")
    end
end

function check_index_consistency(::Tuple{}, ai::IndexSymbols, bi::IndexSymbols)
    rhs_inds = (ai..., bi...)
    if !all(count(k==l for l in rhs_inds) == 2 for k in rhs_inds)
        error("All indices must occur exactly twice on the right-hand side for scalar output")
    end
end

function check_index_consistency(ci::IndexSymbols, ai::IndexSymbols, bi::IndexSymbols)
    rhs_inds = (ai..., bi...)
    # Check that each index occurs only exactly once or twice
    if !all(count(k==l for l in rhs_inds) ∈ (1,2) for k in rhs_inds)
        error("Indices on the right-hand side, i.e. $rhs_inds, can only occur once or twice")
    end
    # Find indices that occurs only once
    free_inds = tuple((k for k in rhs_inds if count(k==l for l in rhs_inds) == 1)...)
    if Set(ci) != Set(free_inds)
        error("The free indices on the lhs ($ci), don't match the free indices on the rhs($(free_inds))")
    end
    # Check that no double indices are given on the lhs
    if !all(count(k==l for l in ci) == 1 for k in ci)
        error("Indices on the left-hand side can only appear once, which is not the case: $ci")
    end
end

function check_input_consistency(C_body, A_head, A_body, B_head, B_body)
    check_arg_expr_consistency(A_head, A_body)
    check_arg_expr_consistency(B_head, B_body)
    check_index_consistency(C_body.inds, A_body.inds, B_body.inds)
end

function get_index_dims(dimA::Int, ai::IndexSymbols, dimB::Int, bi::IndexSymbols)
    if dimA != dimB
        # Will need dispatch for `dimA::Tuple`, `dimB::Tuple` instead for MixedTensor.
        error("This should be fixed for MixedTensor, but for now dims must be equal")
    end
    return dimA
end

get_output_type(ci::Tuple{}, ::Int, ::ArgInfo, ::TermInfo, ::ArgInfo, ::TermInfo) = nothing

function get_output_type(ci::IndexSymbols, dim::Int, Aarg::ArgInfo, Aterm::TermInfo, Barg::ArgInfo, Bterm::TermInfo)
    syminds = Set{Tuple{Symbol,Symbol}}()
    _sort(inds::IndexSymbols{2}) = inds[1] < inds[2] ? inds : (inds[2], inds[1])
    # Collect all sorted pairs of symmetric index names into a Set
    for (arg, term) in ((Aarg, Aterm), (Barg, Bterm))
        arg.type !== :SymmetricTensor && continue
        for i in 2:2:length(term.inds)
            push!(syminds, _sort(term.inds[(i-1):i]))
        end
    end
    # If each consecutive pair of output indices are in syminds, then the output will by symmetric. 
    is_sym = iseven(length(ci)) && all(_sort(ci[(i-1):i]) ∈ syminds for i in 2:2:length(ci))
    return is_sym ? :SymmetricTensor : :Tensor
end

# The following should just return :MixedTensor once supported.
get_output_type(ci::IndexSymbols, dims::NamedTuple, Aarg::ArgInfo, Aterm::TermInfo, Barg::ArgInfo, Bterm::TermInfo) = error("MixedTensor not supported")

function replace_args!(f, args)
    for i in 1:length(args)
        if args[i] isa Symbol
            args[i] = f(args[i])
        elseif args[i] isa Expr
            replace_args!(f, args[i].args)
        end # Could be e.g. LineNumberNode or a number, string etc. 
    end
end

function esc_args!(args; syms=(:A, :B))
    f(s::Symbol) = s ∈ syms ? esc(s) : s
    replace_args!(f, args)
end

function tensor_product!(expr, args...)
    # 1) Analyze the header and function body for information
    # 2) Generate the function body
    # 3) Replace the function body in ´expr` with the generated function body

    # Unpack performance annotations, such as @inbounds and @inline
    if expr.head === :macrocall
        # Check that type is allowed macros (so we don't have to escape)
        @assert expr.args[1] ∈ (Symbol("@inbounds"), Symbol("@inline"))
        @assert length(expr.args) == 3
        @assert expr.args[2] isa LineNumberNode
        tensor_product!(expr.args[3], args...)
        return expr
    elseif expr.head === :tuple # get annotations such as muladd
        if expr.args[1].head !== :function
            error("should be function, but is $(expr.args[1].head)")
        end
        if length(args) != 0
            error("args given in two locations, a: $args, b: $(expr.args[2:end])")
        end
        the_args = expr.args[2:end]
        expr.head = :function
        expr.args = expr.args[1].args
        return tensor_product!(expr, the_args...)
    end

    use_muladd = :muladd in args

    if expr.head !== :function
        error("Unexpected head = $(expr.head)")
    end
    @assert length(expr.args) == 2

    # Header 
    A_arginfo, B_arginfo = extract_header_arginfos(expr.args[1])
    expr.args[1] = esc(expr.args[1]) # Escape header as this should be returned as evaluted in the macro-caller's scope
    
    # Body 
    body = expr.args[2]
    C_terminfo, A_terminfo, B_terminfo = extract_body_information(body)
    check_input_consistency(C_terminfo, A_arginfo, A_terminfo, B_arginfo, B_terminfo)
    dim = get_index_dims(A_arginfo.dim, A_terminfo.inds, B_arginfo.dim, B_terminfo.inds)
    
    # Have checked in extract_body_information that last args in body is the actual expression to be changed.
    # Here, we overwrite the index expression content with the generated expression
    the_expr = get_expression(C_terminfo.inds, A_terminfo.inds, B_terminfo.inds, dim; 
        TC = get_output_type(C_terminfo.inds, dim, A_arginfo, A_terminfo, B_arginfo, B_terminfo), 
        TA = A_arginfo.type, TB = B_arginfo.type, use_muladd
        )
    esc_args!(the_expr.args; syms=(:A, :B))
    body.args[end] = the_expr
    return expr
end

"""
    @tensor_product(expr, args...)

Generate a function to evaluate a tensor product based on an index expression.
```julia 
@tensor_product function my_op(A::Tensor{2,3}, B::Tensor{1,3})
    C[i] = A[i,j]*B[j]
end
```
The type specification of `A` and `B` should contain at least the type of tensor, order, and dim. 
Additional type parameters can optionally be given to dispatch on e.g. the `eltype`.
The return type of `C`, i.e. `Tensor` or `SymmetricTensor` is inferred from the index 
expression and the input tensors.
"""
macro tensor_product(expr, args...)
    tensor_product!(expr, args...)
end

# ======================================================================================
# `@foreach`
# ======================================================================================
function getrange(expr)
    @assert expr.head === :call
    @assert expr.args[1] === :(:)
    @assert all(x->isa(x,Number), expr.args[2:end])
    @assert length(expr.args) ∈ (3,4)
    if length(expr.args) == 3 # from:to range
        return expr.args[2]:expr.args[3]
    else #length(expr.args) == 4 # from:step:to range
        return expr.args[2]:expr.args[3]:expr.args[4]
    end
end

function getiterable(expr)
    if expr.head === :call && expr.args[1] === :(:)
        return getrange(expr)
    elseif expr.head === :(tuple)
        return (a for a in expr.args)
    else
        error("Don't know what to do with $(expr.head)")
    end
end

function loop_over_cases(loopsym, cases, expr)
    exprs = Expr(:tuple)
    for loopvar in getiterable(cases)
        tmpexpr = deepcopy(expr)
        f(s::Symbol) = (s === loopsym ? loopvar : s)
        Tensors.replace_args!(f, tmpexpr.args)
        push!(exprs.args, esc(tmpexpr))
    end
    return exprs
end

function foreach(expr)
    @assert expr.head === :for
    loopsym = expr.args[1].args[1]
    isa(loopsym, Symbol) || error("Can only loop over one variable")
    cases = expr.args[1].args[2]
    codeblock = expr.args[2]::Expr
    @assert codeblock.head === :block
    return loop_over_cases(loopsym, cases, codeblock)
end

"""
    @foreach expr

Given an expression of the form
```julia
for <val> in <range_or_tuple>
    <any code>
end
```
Return one expression for each item in `<range_or_tuple>`, in which all instances of `<val>` 
in `<any code>` is replaced by the value in `<range_or_tuple>`. `<range_or_tuple>` must be
hard-coded. Example 
```julia
@foreach for dim in 1:3
    @foreach for TT in (Tensor, SymmetricTensor)
        Tensors.@tensor_product(@inline @inbounds function my_dot(A::TT{2,dim}, B::TT{2,dim})
            C[i,j] = A[i,k]*B[k,j]
        end)
    end
end
```
"""
macro foreach(expr)
    return foreach(expr)
end
