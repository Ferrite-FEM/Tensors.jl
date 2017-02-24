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
function make_muladd_exp(ex1i, ex2i)
    ex1, ex2 = remove_duplicates(ex1i, ex2i)
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

function remove_duplicates(ex1in, ex2in)
    ex1out, ex2out = Expr[], Expr[]
    exout = Expr[]
    factors = ones(Int, length(ex1in))

    for (ex1ine, ex2ine) in zip(ex1in, ex2in)
        prod = :($ex1ine * $ex2ine)
        i = findfirst(exout, prod) # check if this product exist in the output
        if i == 0 # this product does not exist yet
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
remove_duplicates{order, dim}(::Type{Tensor{order, dim}}, ex) = ex # do nothing if return type is a Tensor

function remove_duplicates{dim}(::Type{SymmetricTensor{2, dim}}, ex)
    if dim == 2
        ex.args = ex.args[[1, 2, 4]]
    elseif dim == 3
        ex.args = ex.args[[1, 2, 3, 5, 6, 9]]
    end
    return ex
end

function remove_duplicates{dim}(::Type{SymmetricTensor{4, dim}}, ex)
    if dim == 2
        ex.args = ex.args[[1, 2, 4, 5, 6, 8, 13, 14, 16]]
    elseif dim == 3
        ex.args = ex.args[[ 1,  2,  3,  5,  6,  9, 10, 11, 12, 14, 15, 18,
                           19, 20, 21, 23, 24, 27, 37, 38, 39, 41, 42, 45,
                           46, 47, 48, 50, 51, 54, 73, 74, 75, 77, 78, 81]]
    end
    return ex
end

# return types
# double contraction
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{4, dim}}, ::Type{Tensor{4, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{4, dim}}, ::Type{SymmetricTensor{4, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{4, dim}}, ::Type{Tensor{4, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{4, dim}}, ::Type{SymmetricTensor{4, dim}}) = SymmetricTensor{4, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{4, dim}}, ::Type{Tensor{2, dim}}) = Tensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{4, dim}}, ::Type{SymmetricTensor{2, dim}}) = Tensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{4, dim}}, ::Type{Tensor{2, dim}}) = SymmetricTensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{4, dim}}, ::Type{SymmetricTensor{2, dim}}) = SymmetricTensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{2, dim}}, ::Type{Tensor{4, dim}}) = Tensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{Tensor{2, dim}}, ::Type{SymmetricTensor{4, dim}}) = SymmetricTensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{2, dim}}, ::Type{Tensor{4, dim}}) = Tensor{2, dim}
getreturntype{dim}(::typeof(dcontract), ::Type{SymmetricTensor{2, dim}}, ::Type{SymmetricTensor{4, dim}}) = SymmetricTensor{2, dim}

# otimes
getreturntype{dim}(::typeof(otimes), ::Type{Tensor{1, dim}}, ::Type{Tensor{1, dim}}) = Tensor{2, dim}
getreturntype{dim}(::typeof(otimes), ::Type{Tensor{2, dim}}, ::Type{Tensor{2, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(otimes), ::Type{SymmetricTensor{2, dim}}, ::Type{Tensor{2, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(otimes), ::Type{Tensor{2, dim}}, ::Type{SymmetricTensor{2, dim}}) = Tensor{4, dim}
getreturntype{dim}(::typeof(otimes), ::Type{SymmetricTensor{2, dim}}, ::Type{SymmetricTensor{2, dim}}) = SymmetricTensor{4, dim}
