#=
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

=#

using Tensors
import Tensors: IndSyms, find_index # new internals used here

# For learning/seeing output
function get_string_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, dims::NamedTuple)
    si = tuple(sort(intersect(ai, bi))...)
    for cinds in Iterators.ProductIterator(tuple((1:dims[k] for k in ci)...))
        print("C[", join(string.(cinds), ", "), "] = ")
        parts = String[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in si)...))
            ainds = tuple((find_index(a, ci, cinds, si, sinds) for a in ai)...)
            binds = tuple((find_index(b, ci, cinds, si, sinds) for b in bi)...)
            push!(parts, string("A[", 
                join(string.(ainds), ", "), "] * B[",
                join(string.(binds), ", "), "]"))
        end
        println(join(parts, " + "))
    end
end
