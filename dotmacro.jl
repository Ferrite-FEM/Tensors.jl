"""
@tensorfun function dcontract(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim    
    C = A[i,j]*B[i,j]
end
@tensorfun function otimes(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim    
    C[i,j,k,l] = A[i,j]*B[k,l]
end
@tensorfun function otimes(A::Tensor{2,2}, B::Tensor{2,2})    
    C[i,j,k,l] = A[i,j]*B[k,l]
end

Restrictions
    * Must be `Tensor{order,dim}` or `SymmetricTensor{order,dim}` (`Vec` not permitted)
    * 


Output, if generic dim given
function \$name(A::Tensor{\$Aorder,dim}, B::Tensor{\$Border,dim}) where dim 
    if dim == 1
        return \$expr_1d
    elseif dim == 2
        return \$expr_2d
    else
        return \$expr_3d
    end
end
if a specific dim given 
function \$name(A::Tensor{\$Aorder,dim}, B::Tensor{\$Border,dim}) where dim 
    return \$expr
end

"""
const TupleN{T,N} = NTuple{N,T}

function foo(c::TupleN{Symbol}, a::TupleN{Symbol}, b::TupleN{Symbol}; use_muladd=false)
    # checks
    all(allunique.((a,b,c))) || error("a, b, or c has duplicate indices unique: a=", a, ", b=", b, ", c=", c)
    all_indicies = union(a,b)
    dummy_indicies = intersect(a,b)
    free_indices = setdiff(all_indicies, dummy_indicies)
    Set(c) == Set(free_indices) || error("The free indices on the lhs, ", c, " are not matching the rhs, ", free_indices)
end

function create_op_loop(c, a, b)
    expr = Expr(:for)
    # loop header 
    expr_head = Expr(:block)
    for ind in reverse(c)
        push!(expr_head.args, Expr(:(=), ind, :(1:dim)))
    end
    push!(expr.args, expr_head)
    # loop body 
    expr_body = Expr(:block)
    
end

function build_expression(TA, TB, c, a, b, muladd=false)
    idxA(args...) = compute_index(get_base(S1), args...)
    idxB(args...) = compute_index(get_base(S2), args...)
end 

e = Meta.parse("for j in 1:dim, i in 1:dim; println(i); println(j); end")

ef = Expr(:for)
ef_inds = Expr(:block)
c = (:i, :j)
for ind in c
    push!(ef_inds.args, Expr(:(=), c, :(1:dim)))
end
push!(ef.args, ef_inds)

ef_body = Expr(:block)

#=
    indf(inds) = join(string.(inds), ",")
    loop_vals(s::Symbol) = join((s, " in 1:dim"))
    loop_string(inds) = "for "*join(loop_vals.(inds), ",")
    exp_builder(S, inds) = join(("ex", S, " = Expr[:(get_data(", S, ")[(\$(idx", S, "(", indf(inds), ")))])", " for ", loop_string(dummy_indicies), "][:]"))

    idxA = Meta.parse(join(("idxA(", indf(a), ") = compute_index(get_base(A),", indf(a), ")")))
    idxB = Meta.parse(join(("idxB(", indf(b), ") = compute_index(get_base(B),", indf(b), ")")))
    loop = #Meta.parse(
        join((loop_string(reverse(c)),";", 
            "println(", c, ");",
            "println(", free_indices, ");",
            #exp_builder(:A, a), ";", 
            #exp_builder(:B, b), ";",
            #"push!(exps.args, reducer(exA, exB, ", use_muladd, "));",
            "end"))
        #)
=#