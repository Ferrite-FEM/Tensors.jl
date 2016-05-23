
function inline_body(T)
    if T == Float64 || T == Float32
        return Expr(:meta, :inline)
    else
        return Expr(:tuple)
    end
end

tensor_create{order, dim}(::Type{Tensor{order, dim}}, f) = tensor_create(Tensor{order, dim, Float64}, f)
function tensor_create{order, dim, T}(::Type{Tensor{order, dim, T}}, f)
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dim, j=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
    end
    return quote
        $ex
    end
end

tensor_create_no_arg{order, dim}(::Type{Tensor{order, dim}}, f) = tensor_create_no_arg(Tensor{order, dim, Float64}, f)
function tensor_create_no_arg{order, dim, T}(::Type{Tensor{order, dim, T}}, f)
    if order == 1
        ex = Expr(:tuple, [f() for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f() for i=1:dim, j=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f() for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
    end
    return quote
        $ex
    end
end

tensor_create_elementwise{order, dim}(::Type{Tensor{order, dim}}, f) = tensor_create_elementwise(Tensor{order, dim, Float64}, f)
function tensor_create_elementwise{order, dim, T}(::Type{Tensor{order, dim, T}}, f)
    z = 0
    if order == 1
        ex = Expr(:tuple, [f(z+=1) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(z+=1) for i=1:dim, j=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f(z+=1) for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
    end
    return quote
        $ex
    end
end

@generated function to_tuple{N}(::Type{NTuple{N}},  data::AbstractArray)
    return Expr(:tuple, [:(data[$i]) for i=1:N]...)
end

@inline to_tuple{N}(::Type{NTuple{N}},  data::NTuple{N}) = data
