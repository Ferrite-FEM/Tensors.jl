tensor_create{order, dim}(::Type{SymmetricTensor{order, dim}}, f) = tensor_create(SymmetricTensor{order, dim, Float64}, f)
function tensor_create{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, f)
    expr = Any[]
    if order == 2
        for i in 1:dim, j in i:dim
            push!(expr, f(j, i))
        end
    elseif order == 4
        for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
            push!(expr, f(j, i, l, k))
        end
    end
    return quote
        $(Expr(:tuple, expr...))
    end
end

@generated function dot_matmatT{N, T}(F::NTuple{N, T}, i, j)
    rows = Int(sqrt(N))
    return quote
        $(Expr(:meta, :inline))
        s = zero(T)
        @inbounds for k = 1:$rows
            s += mat_get_index(F, k, i) * mat_get_index(F, k, j)
        end
        s
    end
end

@generated function transpdot{N}(A::NTuple{N})
    rows = Int(sqrt(N))
    exps = Vector{Expr}()
    for i in 1:rows, j in 1:(rows-i+1)
        push!(exps, :(dot_matmatT(A, $(j + i -1), $i)))
    end
    body = Expr(:tuple, exps...)
    return quote
       $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end
