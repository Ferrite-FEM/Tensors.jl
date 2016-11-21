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
        for i in 1:dim, j in i:dim
            push!(ex.args, f(i, j))
        end
    elseif order == 4
        for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
            push!(ex.args, f(i, j, k, l))
        end
    end
    return ex
end
