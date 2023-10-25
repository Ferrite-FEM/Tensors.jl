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
        i = findfirst(isequal(prod), exout) # check if this product exist in the output
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

# unit stripping if necessary
function ustrip(S::SymmetricTensor{order,dim,T}) where {order, dim, T}
    ou = oneunit(T)
    if typeof(ou / ou) === T # no units
        return S
    else # units, so strip them by dividing with oneunit(T)
        return SymmetricTensor{order,dim}(map(x -> x / ou, S.data))
    end
end

# dotmacro
#= Aiming for a syntax in the style of 
@tensorfun function dcontract(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim
    C = A[i,j]*B[i,j]
end
@tensorfun function otimes(A::Tensor{2,dim}, B::Tensor{2,dim}) where dim    
    C[i,j,k,l] = A[i,j]*B[k,l]
end
@tensorfun function otimes(A::Tensor{2,2}, B::Tensor{2,2})    
    C[i,j,k,l] = A[i,j]*B[k,l]
end
=#

const IndSyms{N} = NTuple{N,Symbol}
const IntVals{N} = NTuple{N,Int}

function find_index(ind::Symbol, ci::IndSyms, cinds::IntVals, si::IndSyms, sinds::IntVals)
    i = findfirst(Base.Fix1(===, ind), ci)
    i !== nothing && return cinds[i]
    i = findfirst(Base.Fix1(===, ind), si)
    i !== nothing && return sinds[i]
    error("Could not find $ind in ci=$ci or si=$si")
end

function get_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, dim::Int; kwargs...)
    indsyms = union(ci, ai, bi)
    dims = NamedTuple(k=>dim for k in indsyms)
    return get_expression(ci, ai, bi, dims; kwargs...)
end

"""
    get_expression(ci, ai, bi, dims; kwargs...)

Example to calculate `C[i,j] = A[i,l,m]*B[l,m,j]` in `dim=2`
```julia
get_expression((:i, :j), (:i, :l, :m), (:l, :m, :j), 2)
```
"""
function get_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, 
        dims::NamedTuple;
        #idxA::Function, idxB::Function,
        TC=Tensor, TA=Tensor, TB=Tensor,
        use_muladd=false)
    @assert allequal(values(dims)) # Only required for correct idxA/B funs, to be changed for mixed tensors
    dim = first(values(dims))
    idxA(args...) = compute_index(TA{length(ai),dim}, args...)
    idxB(args...) = compute_index(TB{length(bi),dim}, args...)
    TensorType = TC{length(ci),dim}
    si = tuple(sort(intersect(ai, bi))...)
    exps = Expr(:tuple)
    for cinds in Iterators.ProductIterator(tuple((1:dims[k] for k in ci)...))
        exa = Expr[]
        exb = Expr[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in si)...))
            ainds = tuple((find_index(a, ci, cinds, si, sinds) for a in ai)...)
            binds = tuple((find_index(b, ci, cinds, si, sinds) for b in bi)...)
            push!(exa, :(get_data(A)[$(idxA(ainds...))]))
            push!(exb, :(get_data(B)[$(idxB(binds...))]))
        end
        push!(exps.args, reducer(exa, exb, use_muladd))
    end
    return remove_duplicates(TensorType, exps)
end