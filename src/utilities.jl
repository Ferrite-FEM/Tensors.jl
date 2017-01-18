function tensor_create{order, dim}(::Type{Tensor{order, dim}}, f)
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dim, j=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
    end
    return ex
end

function tensor_create{order, dim}(::Type{SymmetricTensor{order, dim}}, f)
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

# create recursive muladd exp from two expression arrays
function make_muladd_exp(ex1, ex2)
    N = length(ex1)
    ex = Expr(:call)
    exn = Expr(:call, :*, ex1[1], ex2[1])

    if N == 1 # return only the multiplication
        return exn
    end

    for i in 2:N
        ex = Expr(:call, :muladd)
        push!(ex.args, ex1[i])
        push!(ex.args, ex2[i])
        push!(ex.args, exn)
        exn = ex
    end
    return ex
end
