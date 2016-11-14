@generated function mat_get_index{N}(t::NTuple{N}, i::Int, j::Int)
    rows = Int(sqrt(N))
    return quote
        $(Expr(:meta, :inline))
        @inbounds v = t[(j-1) * $rows + i]
        return v
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


@generated function to_tuple{N}(::Type{NTuple{N}},  data::AbstractArray)
    return Expr(:tuple, [:(data[$i]) for i=1:N]...)
end

@inline to_tuple{N}(::Type{NTuple{N}},  data::NTuple{N}) = data
