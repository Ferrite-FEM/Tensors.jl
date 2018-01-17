function tensor_create_linear(T::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}, f) where {order, dim}
    return Expr(:tuple, [f(i) for i=1:n_components(T)]...)
end

function tensor_create(::Type{Tensor{order, dim}}, f) where {order, dim}
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dim, j=1:dim]...)
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
        i = findfirst(equalto(prod), exout) # check if this product exist in the output
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

# otimes
function otimes end
@pure getreturntype(::typeof(otimes), ::Type{<:Tensor{1, dim}}, ::Type{<:Tensor{1, dim}}) where {dim} = Tensor{2, dim}
@pure getreturntype(::typeof(otimes), ::Type{<:SecondOrderTensor{dim}}, ::Type{<:SecondOrderTensor{dim}}) where {dim} = Tensor{4, dim}
@pure getreturntype(::typeof(otimes), ::Type{<:SymmetricTensor{2, dim}}, ::Type{<:SymmetricTensor{2, dim}}) where {dim} = SymmetricTensor{4, dim}
