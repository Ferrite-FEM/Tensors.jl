function tensor_create_linear(T::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}, f) where {order, dim}
    return Expr(:tuple, [f(i) for i=1:n_components(T)]...)
end

function tensor_create(::Type{Tensor{order, dim}}, f) where {order, dim}
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dim, j=1:dim]...)
    elseif order == 3
        ex = Expr(:tuple, [f(i,j,k) for i=1:dim, j=1:dim, k=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
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
function remove_duplicates(::Type{SymmetricTensor{order, dim}}, ex) where {order, dim}
    ex.args = ex.args[SYMMETRIC_INDICES[order][dim]]
    return ex
end

# return types
# double contraction
function dcontract end
@pure getreturntype(::typeof(dcontract), ::Type{<:FourthOrderTensor{dim}}, ::Type{<:FourthOrderTensor{dim}}) where {dim} = Tensor{4, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:SymmetricTensor{4, dim}}, ::Type{<:SymmetricTensor{4, dim}}) where {dim} = SymmetricTensor{4, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:Tensor{4, dim}}, ::Type{<:SecondOrderTensor{dim}}) where {dim} = Tensor{2, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:SymmetricTensor{4, dim}}, ::Type{<:SecondOrderTensor{dim}}) where {dim} = SymmetricTensor{2, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:SecondOrderTensor{dim}}, ::Type{<:Tensor{4, dim}}) where {dim} = Tensor{2, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:SecondOrderTensor{dim}}, ::Type{<:SymmetricTensor{4, dim}}) where {dim} = SymmetricTensor{2, dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:Tensor{3,dim}}, ::Type{<:SecondOrderTensor{dim}}) where {dim} = Vec{dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:SecondOrderTensor{dim}}, ::Type{<:Tensor{3,dim}}) where {dim} = Vec{dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:Tensor{3,dim}}, ::Type{<:FourthOrderTensor{dim}}) where {dim} = Tensor{3,dim}
@pure getreturntype(::typeof(dcontract), ::Type{<:FourthOrderTensor{dim}}, ::Type{<:Tensor{3,dim}}) where {dim} = Tensor{3,dim}

# otimes
function otimes end
@pure getreturntype(::typeof(otimes), ::Type{<:Tensor{1, dim}}, ::Type{<:Tensor{1, dim}}) where {dim} = Tensor{2, dim}
@pure getreturntype(::typeof(otimes), ::Type{<:SecondOrderTensor{dim}}, ::Type{<:SecondOrderTensor{dim}}) where {dim} = Tensor{4, dim}
@pure getreturntype(::typeof(otimes), ::Type{<:SymmetricTensor{2, dim}}, ::Type{<:SymmetricTensor{2, dim}}) where {dim} = SymmetricTensor{4, dim}

# unit stripping if necessary
function ustrip(S::SymmetricTensor{order,dim,T}) where {order, dim, T}
    ou = oneunit(T)
    if typeof(ou / ou) === T # no units
        return S
    else # units, so strip them by dividing with oneunit(T)
        return SymmetricTensor{order,dim}(map(x -> x / ou, S.data))
    end
end

# dotmacro
#= Aiming for a syntax in the style of 
@tensorfun function dcontract(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim
    C = A[i,j]*B[i,j]
end
@tensorfun function otimes(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim    
    C[i,j,k,l] = A[i,j]*B[k,l]
end
@tensorfun function otimes(A::Tensor{2,2}, B::Tensor{2,2})    
    C[i,j,k,l] = A[i,j]*B[k,l]
end
=#

const IndSyms{N} = NTuple{N,Symbol}
const IntVals{N} = NTuple{N,Int}

function find_index(ind::Symbol, si::IndSyms, sinds::IntVals)
    i = findfirst(Base.Fix1(===, ind), si)
    i !== nothing && return sinds[i]
    error("Could not find $ind in si=$si")
end

function find_index(ind::Symbol, ci::IndSyms, cinds::IntVals, si::IndSyms, sinds::IntVals)
    i = findfirst(Base.Fix1(===, ind), ci)
    i !== nothing && return cinds[i]
    i = findfirst(Base.Fix1(===, ind), si)
    i !== nothing && return sinds[i]
    error("Could not find $ind in ci=$ci or si=$si")
end

"""
    get_expression(ci, ai, bi, dims; kwargs...)

Examples to get the expression for the following with `dim=2`
* `C = A[i]*B[i]`
* `C[i] = A[i,j]*B[j]`
* `C[i,j] = A[i,l,m]*B[l,m,j]` 
```julia
get_expression((), (:i,), (:i,), 2)
get_expression((:i,), (:i, :j), (:j,), 2)
get_expression((:i, :j), (:i, :l, :m), (:l, :m, :j), 2)
```
"""
function get_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, 
        dims::NamedTuple; TC, TA, TB, use_muladd=false
        )
    
    idxA(args...) = compute_index(TA, args...)
    idxB(args...) = compute_index(TB, args...)

    si = tuple(sort(intersect(ai, bi))...)
    exps = Expr(:tuple)
    for cinds in Iterators.ProductIterator(tuple((1:dims[k] for k in ci)...))
        exa = Expr[]
        exb = Expr[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in si)...))
            ainds = tuple((find_index(a, ci, cinds, si, sinds) for a in ai)...)
            binds = tuple((find_index(b, ci, cinds, si, sinds) for b in bi)...)
            push!(exa, :(get_data(A)[$(idxA(ainds...))]))
            push!(exb, :(get_data(B)[$(idxB(binds...))]))
        end
        push!(exps.args, reducer(exa, exb, use_muladd))
    end
    return :($TC($(remove_duplicates(TC, exps))))
end

function get_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, dim::Int; kwargs...)
    indsyms = union(ci, ai, bi)
    dims = NamedTuple(k=>dim for k in indsyms)
    return get_expression(ci, ai, bi, dims; kwargs...)
end

# For scalar output
function get_expression(::Tuple{}, ai::IndSyms, bi::IndSyms, 
        dims::NamedTuple; TC::Nothing, TA, TB, use_muladd=false
        )
    
    idxA(args...) = compute_index(TA, args...)
    idxB(args...) = compute_index(TB, args...)
    
    si = tuple(sort(intersect(ai, bi))...)
    exa = Expr[]
    exb = Expr[]
    for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in si)...))
        ainds = tuple((find_index(a, si, sinds) for a in ai)...)
        binds = tuple((find_index(b, si, sinds) for b in bi)...)
        push!(exa, :(get_data(A)[$(idxA(ainds...))]))
        push!(exb, :(get_data(B)[$(idxB(binds...))]))
    end
    return reducer(exa, exb, use_muladd)
end

function get_expression(ci::Tuple{}, ai::IndSyms, bi::IndSyms, dim::Int; kwargs...)
    indsyms = union(ai, bi)
    dims = NamedTuple(k=>dim for k in indsyms)
    return get_expression(ci, ai, bi, dims; kwargs...)
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
    @assert type in (:Tensor, #=SymmetricTensor, MixedTensor=#)
    order = curly.args[2]::Int # Use type-assert as sanity check
    dim = curly.args[3]::Int   # Use type-assert as sanity check
    basetype = :($type{$order, $dim})
    return (name=name, type=type, order=order, dim=dim, basetype)
end

function extract_header_information(header::Expr)
    header.head === :call || error("header expression with head=$(header.head) is not supported")
    fname = header.args[1]
    Ainfo = extract_arginfo(header.args[2])
    Binfo = extract_arginfo(header.args[3])
    return fname, Ainfo, Binfo
end

function extract_terminfo(term::Expr)
    @assert term.head === :ref
    name = term.args[1]
    inds = tuple(term.args[2:end]...)
    return (name=name, inds=inds)
end

# Scalar term, no indices
function extract_terminfo(term::Symbol)
    return (name=term, inds=())
end

function extract_body_information(body::Expr)
    body.head === :block || error("Expecting a block type expression")
    @assert all(isa(a, LineNumberNode) for a in body.args[1:(length(body.args)-1)])
    @assert body.args[end].head === :(=) # Should be an assignment
    expr = body.args[end].args
    Cinfo = extract_terminfo(expr[1])
    @assert expr[2].head === :call
    @assert expr[2].args[1] === :*
    Ainfo = extract_terminfo(expr[2].args[2])
    Binfo = extract_terminfo(expr[2].args[3])
    return Cinfo, Ainfo, Binfo
end

function check_arg_expr_consistency(head, body)
    head.name === body.name || error("head ($head) and body ($body) variable names don't match")
    if length(body.inds) !== head.order
        ninds = length(body.inds)
        error("head for $(head.name) specifices tensor of order $(head.order), but index expression has only $ninds ($(body.inds))")
    end
end

function check_index_consistency(::Tuple{}, ai::IndSyms, bi::IndSyms)
    rhs_inds = (ai..., bi...)
    if !all(count(k==l for l in rhs_inds) == 2 for k in rhs_inds)
        error("All indices must occur exactly twice on the right-hand side for scalar output")
    end
end

function check_index_consistency(ci::IndSyms, ai::IndSyms, bi::IndSyms)
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

function get_index_dims(dimA::Int, ai::IndSyms, dimB::Int, bi::IndSyms)
    if dimA != dimB
        # Will need dispatch for `dimA::Tuple`, `dimB::Tuple` instead then.
        error("This should be fixed for MixedTensor, but for now dims must be equal")
    end
    return dimA
end

get_output_type(ci::Tuple{}, dim::Int, args...) = nothing
function get_output_type(ci::IndSyms, dim::Int, A_headinfo, A_bodyinfo, B_headinfo, B_bodyinfo)
    # Should support SymmetricTensors and MixedTensor in the future
    # For MixedTensor, dim::NamedTuple must be supported.
    @assert A_headinfo.type === :Tensor
    @assert B_headinfo.type === :Tensor
    return Tensor{length(ci), dim}
end

function esc_args!(args; syms=(:A, :B))
    for i in 1:length(args)
        if args[i] isa Symbol
            args[i] ∈ syms && (args[i] = esc(args[i]))
        elseif args[i] isa Expr
            esc_args!(args[i].args; syms)
        end # Could be e.g. line numbers
    end
end

function tensor_product!(expr, args...)
    use_muladd = :muladd in args

    @assert expr.head === :function
    @assert length(expr.args) == 2
    # Header 
    fname, A_headinfo, B_headinfo = extract_header_information(expr.args[1])
    expr.args[1] = esc(expr.args[1]) # Escape header as this should be returned as evaluted in the macro-caller's scope
    # Body 
    body = expr.args[2]
    C_bodyinfo, A_bodyinfo, B_bodyinfo = extract_body_information(body)
    check_input_consistency(C_bodyinfo, A_headinfo, A_bodyinfo, B_headinfo, B_bodyinfo)
    dim = get_index_dims(A_headinfo.dim, A_bodyinfo.inds, B_headinfo.dim, B_bodyinfo.inds)
    TC = get_output_type(C_bodyinfo.inds, dim, A_headinfo, A_bodyinfo, B_headinfo, B_bodyinfo)
    # Not sure how to avoid eval here...
    TA = eval(A_headinfo.basetype)
    TB = eval(B_headinfo.basetype) 
    # Have checked in extract_body_information that last args in body is the actual expression to be changed.
    # Here, we overwrite the index expression content with the generated expression
    the_expr = get_expression(C_bodyinfo.inds, A_bodyinfo.inds, B_bodyinfo.inds, dim; TC, TA, TB, use_muladd)
    esc_args!(the_expr.args; syms=(:A, :B))
    body.args[end] = the_expr
    return expr
end

macro tensor_product(expr, args...)
    # Plan
    # 1) Analyze the header for information
    # 2) Analyze the function body for information
    # 3) Given this information, generate the function body
    # 4) Return the expression in which the function body has been replaced
    #    by the generated expression, keeping the function header intact.
    #    This opens up, for example, the possibility of using different performance
    #    annotations for different datatypes.
    tensor_product!(expr, args...)
end

# Generate a few test cases, just to check that it works.
function m_dcontract end
function m_otimes end
function m_dot end

@tensor_product (function m_dcontract(A::Tensor{2,3}, B::Tensor{2,3})
    C = A[i,j]*B[i,j]
end)

@tensor_product (function m_otimes(A::Tensor{1,3}, B::Tensor{1,3})
    C[i,j] = A[i]*B[j]
end)

@tensor_product (function m_dot(A::Tensor{2,3}, B::Tensor{2,3})
    C[i,j] = A[i,k]*B[k,j]
end)