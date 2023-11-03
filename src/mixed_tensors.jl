import Base.@pure

export MixedTensor
# A mixed tensor doesn't have the same dimension for each leg 
# This might be useful for Ferrite when using special cellvalues 
# where the dimensions of the coordinate and shape function is not 
# the same (and shape function is not scalar)
struct MixedTensor{order, dims, T, M} <: AbstractTensor{order, dims, T}
    data::NTuple{M, T}
    function MixedTensor{order, dims, T, M}(data::NTuple) where {order, dims, T, M}
        # Temporary checks for developing
        @assert isa(dims, NTuple{order,Int}) # order isn't actually required, but good for dispatch
        @assert prod(dims) == M              # n_components must be correct
        return new{order, dims, T, M}(data)
    end
end

# Steal base implementation of "prod" to safely mark with @pure 
@pure n_components(::Type{MixedTensor{order, dims}}) where {order, dims} = *(dims...)

@pure get_base(::Type{<:MixedTensor{order, dims}}) where {order, dims} = MixedTensor{order, dims}

@pure Base.eltype(::Type{MixedTensor{order, dims, T, M}}) where {order, dims, T, M} = T
@pure Base.eltype(::Type{MixedTensor{order, dims, T}})    where {order, dims, T}    = T
@pure Base.eltype(::Type{MixedTensor{order, dims}})       where {order, dims}       = Any

############################
# Abstract Array interface #
############################
Base.IndexStyle(::Type{<:MixedTensor}) = IndexLinear()

########
# Size #
########
Base.size(::MixedTensor{<:Any, dims}) where dims = dims
Base.length(::Type{MixedTensor{<:Any, <:Any, <:Any, M}}) where M = M

#########################
# Internal constructors #
#########################
function dims_permutations(order, maxdim=3)
    # Get all permutations for the given order 
    # e.g. for order=2 we have (1,1), (1,2), (1,3), (2,1), (2,2), (2,3), (3,1), (3,2), (3,3)
    if order == 1
        return ntuple(i->(i,), maxdim)
    elseif order == 2
        return tuple(((i,j) for i in 1:maxdim, j in 1:maxdim)...)
    elseif order == 3
        return tuple(((i,j,k) for i in 1:maxdim, j in 1:maxdim, k in 1:maxdim)...)
    elseif order == 4
        return tuple(((i,j,k,l) for i in 1:maxdim, j in 1:maxdim, k in 1:maxdim, l in 1:maxdim)...)
    else
        throw(ArgumentError("order=$order not supported"))
    end
end

for order in (1,2,3,4)
    for dims in dims_permutations(order)
        M = n_components(MixedTensor{order, dims})
        @eval begin
            @inline MixedTensor{$order, $dims}(t::NTuple{$M, T}) where T = MixedTensor{$order, $dims, T, $M}(t)
            @inline MixedTensor{$order, $dims, T}(t::NTuple{$M}) where T = MixedTensor{$order, $dims, T, $M}(t)
        end
        if M > 1 # To avoid overwriting ::Tuple{Any}
            # Heterogeneous tuple
            @eval @inline MixedTensor{$order, $dims}(t::Tuple{Vararg{<:Any,$M}}) = MixedTensor{$order, $dims}(promote(t...))
        end
    end
end

## Indexing 
@inline function Base.getindex(S::MixedTensor, i::Int)
    @boundscheck checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

## Create from function 
function tensor_create_linear(T::Type{MixedTensor{order, dims}}, f) where {order, dims}
    return Expr(:tuple, [f(i) for i=1:n_components(T)]...)
end

function tensor_create(::Type{MixedTensor{order, dims}}, f) where {order, dims}
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dims[1]]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dims[1], j=1:dims[2]]...)
    elseif order == 3
        ex = Expr(:tuple, [f(i,j,k) for i=1:dims[1], j=1:dims[2], k=1:dims[3]]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dims[1], j=1:dims[2], k = 1:dims[3], l = 1:dims[4]]...)
    end
    return ex
end

@generated function (S::Type{MixedTensor{order, dims}})(f::Function) where {order, dims}
    TensorType = get_base(get_type(S))
    if order == 1
        exp = tensor_create(TensorType, (i) -> :(f($i)))
    elseif order == 2
        exp = tensor_create(TensorType, (i,j) -> :(f($i, $j)))
    elseif order == 4
        exp = tensor_create(TensorType, (i,j,k,l) -> :(f($i, $j, $k, $l)))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

# Applies the function f to all indices f(1), f(2), ... f(n_independent_components)
@generated function apply_all(S::Type{MixedTensor{order, dims}}, f::Function) where {order, dims}
    TensorType = get_base(get_type(S))
    exp = tensor_create_linear(TensorType, (i) -> :(f($i)))
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

@inline function apply_all(S::MixedTensor{order, dims}, f::Function) where {order, dims}
    apply_all(get_base(typeof(S)), f)
end

## Basic operations

# Unary
@inline Base.:+(S::MixedTensor) = S
@inline Base.:-(S::MixedTensor) = _map(-, S)

# Binary
@inline Base.:+(S1::MixedTensor{order, dims}, S2::MixedTensor{order, dims}) where {order, dims} = _map(+, S1, S2)
@inline Base.:-(S1::MixedTensor{order, dims}, S2::MixedTensor{order, dims}) where {order, dims} = _map(-, S1, S2)

@inline Base.:*(S::MixedTensor, n::Number) = _map(x -> x*n, S)
@inline Base.:*(n::Number, S::MixedTensor) = _map(x -> n*x, S)
@inline Base.:/(S::MixedTensor, n::Number) = _map(x -> x/n, S)

# map implementations
@inline function _map(f, S1::MixedTensor{order,dims}, S2::MixedTensor{order,dims}) where {order, dims}
    return apply_all(S1, @inline function(i) @inbounds f(S1.data[i], S2.data[i]); end)
end

# Convert to regular tensor if possible
# isregular required for type stability
isregular(::MixedTensor{1}) = true
isregular(::MixedTensor{2,dims}) where dims = dims[1]==dims[2]
isregular(::MixedTensor{4,dims}) where dims = dims[1]==dims[2]==dims[3]==dims[4]

function makeregular(t::MixedTensor{order,dims}) where {order,dims}
    if isregular(t)
        return Tensor{order,dims[1]}(get_data(t))
    else
        return t
    end
end
makemixed(t::Tensor{1,dim}) where dim = MixedTensor{1,(dim,)}(get_data(t))
makemixed(t::Tensor{2,dim}) where dim = MixedTensor{2,(dim,dim)}(get_data(t))
makemixed(t::Tensor{4,dim}) where dim = MixedTensor{4,(dim,dim,dim,dim)}(get_data(t))


# Slow (not all) implementations just for testing 
dcontract(S1::MixedTensor, S2::Tensor) = dcontract(S1, makemixed(S2))
dcontract(S1::Tensor, S2::MixedTensor) = dcontract(makemixed(S1), S2)
function dcontract(S1::MixedTensor{2,dims}, S2::MixedTensor{2,dims}) where dims
    mapreduce(*, +, get_data(S1), get_data(S2))
end

function dcontract(S1::MixedTensor{4,dims1}, S2::MixedTensor{2,dims2}) where {dims1, dims2}
    dims1[3:4] == dims2 || throw(DimensionMismatch("$dims1, $dims2"))
    I, J, K, L = dims1
    makeregular(
        MixedTensor{2,(I,J)}(
            (i,j)->sum(kl->S1[i,j,kl[1],kl[2]]*S2[kl[1],kl[2]],
                ntuple(m-> (rem(m-1,K)+1,div(m-1,K)+1), K*L)
                )))
end

function dcontract(S1::MixedTensor{2,dims1}, S2::MixedTensor{4,dims2}) where {dims1, dims2}
    dims1 == dims2[1:2] || throw(DimensionMismatch("$dims1, $dims2"))
    I, J, K, L = dims2
    makeregular(
        MixedTensor{2,(K,L)}(
            (k,l)->sum(ij->S1[ij[1], ij[2]]*S2[ij[1],ij[2], k, l],
                ntuple(m-> (rem(m-1,I)+1,div(m-1,I)+1), I*J)
                )))
end

function dcontract(S1::MixedTensor{4,dims1}, S2::MixedTensor{4,dims2}) where {dims1, dims2}
    dims1[3:4] == dims2[1:2] || throw(DimensionMismatch("$dims1, $dims2"))
    I, J, K, L = dims1
    M, N = dims2[3:4]
    makeregular(
        MixedTensor{4, (I,J,M,N)}(
        (i,j,m,n) -> sum(kl->S1[i,j,kl[1],kl[2]]*S2[kl[1],kl[2],m,n],
        ntuple(o-> (rem(o,K)+1,div(o-1,K)+1), K*L)
        )))
end

otimes(S1::Tensor, S2::MixedTensor) = otimes(makemixed(S1), S2)
otimes(S1::MixedTensor, S2::Tensor) = otimes(S1, makemixed(S2))

function otimes(S1::Vec{d1}, S2::Vec{d2}) where {d1, d2}
    return MixedTensor{2, (d1, d2)}(@inline function(i,j) @inbounds S1[i]*S2[j]; end)
end

function otimes(S1::Tensor{2,d1}, S2::Tensor{2,d2}) where {d1, d2}
    return MixedTensor{4, (d1, d1, d2, d2)}(
        @inline function(i,j,k,l) 
            return @inbounds S1[i,j]*S2[k,l]
        end)
end

function otimes(S1::MixedTensor{2,dims1}, S2::MixedTensor{2,dims2}) where {dims1, dims2}
    return makeregular(
        MixedTensor{4, (dims1[1], dims1[2], dims2[1], dims2[2])}(
        @inline function(i,j,k,l)
            return @inbounds S1[i,j]*S2[k,l]
        end))
end

function LinearAlgebra.dot(S1::MixedTensor{1,dims}, S2::MixedTensor{1,dims}) where dims 
    return mapreduce(*, +, get_data(S1), get_data(S2))
end

function LinearAlgebra.dot(S1::MixedTensor{1,dims1}, S2::MixedTensor{2,dims2}) where {dims1, dims2}
    dims1[1] == dims2[1] || throw(ArgumentError("$dims1, $dims2"))
    return Vec{dims2[2]}(j -> sum(i->S1[i]*S2[i,j], 1:dims1[1]))
end

function LinearAlgebra.dot(S1::MixedTensor{2,dims1}, S2::MixedTensor{1,dims2}) where {dims1, dims2}
    dims1[2] == dims2[1] || throw(ArgumentError("$dims1, $dims2"))
    return Vec{dims1[1]}(i -> sum(j->S1[i,j]*S2[j], 1:dims2[1]))
end

function LinearAlgebra.dot(S1::MixedTensor{2,dims1}, S2::MixedTensor{2,dims2}) where {dims1, dims2}
    dims1[2] == dims2[1] || throw(ArgumentError("$dims1, $dims2"))
    return makeregular(MixedTensor{2, (dims1[1], dims2[2])}((i,k)->sum(j->S1[i,j]*S2[j,k], 1:dims1[2])))
end

# TODO: 4th order dot products

# Lazy for now, depends on performance of mixed tensors if this is fine. 
LinearAlgebra.dot(S1::MixedTensor, S2::Tensor) = dot(S1, makemixed(S2))
LinearAlgebra.dot(S1::Tensor, S2::MixedTensor) = dot(makemixed(S1), S2)

@inline function Base.transpose(S::MixedTensor{2, dims}) where {dims}
    MixedTensor{2, (dims[2],dims[1])}(@inline function(i, j) @inbounds S[j,i]; end)
end

@inline Base.adjoint(S::MixedTensor) = transpose(S)