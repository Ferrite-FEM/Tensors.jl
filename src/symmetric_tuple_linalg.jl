function sym_tupexpr_mat(f,N)
    rows = Int( div(sqrt(1 + 8*N), 2))
    expr = Any[]
    for i in 1:rows, j in 1:(rows-i+1)
        push!(expr, f(j+i-1,i))
    end
    return quote
        @inbounds return $(Expr(:tuple, expr...))
    end
end

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

tensor_create_no_arg{order, dim}(::Type{SymmetricTensor{order, dim}}, f) = tensor_create_no_arg(SymmetricTensor{order, dim, Float64}, f)
function tensor_create_no_arg{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, f)
    expr = Any[]
    if order == 2
        for i in 1:dim, j in i:dim
            push!(expr, f())
        end
    elseif order == 4
        for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
            push!(expr, f())
        end
    end
    return quote
        $(Expr(:tuple, expr...))
    end
end

tensor_create_elementwise{order, dim}(::Type{SymmetricTensor{order, dim}}, f) = tensor_create_elementwise(SymmetricTensor{order, dim, Float64}, f)

function tensor_create_elementwise{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, f)
    expr = Any[]
    z = 0
    if order == 1
        for i in 1:dim
            push!(expr, f(z+=1))
    end
    elseif order == 2
        for i in 1:dim, j in i:dim
            push!(expr, f(z+=1))
        end
    elseif order == 4
        for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
            push!(expr, f(z+=1))
        end
    end
    return quote
        $(Expr(:tuple, expr...))
    end
end

function sym_mat_get_index(N::Int, i::Int, j::Int)
    dim = Int( div(sqrt(1 + 8*N), 2))
    skipped_indicies = div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@generated function sym_mat_set_index{N, T, I, J}(a::NTuple{N, T}, v, ::Type{Val{I}}, ::Type{Val{J}})
    dim = Int( div(sqrt(1 + 8*N), 2))
    b = sym_tupexpr_mat((i,j) -> (i == I && j == J) ? :(v) : :(a[$(sym_mat_get_index(N, i, j))]), N)
    return quote
        $(inline_body(T))
        $b
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
