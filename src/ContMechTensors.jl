#__precompile__()

module ContMechTensors

immutable InternalError <: Exception end

export SymmetricTensor, Tensor, Vec

export otimes, otimes_unsym, otimes_unsym!, otimes!, ⊗, ⊠, dcontract, dev, dev!
export extract_components, load_components!

#########
# Types #
#########
abstract AbstractTensor{order, dim, T <: Number, M} <: AbstractArray{T, order}

immutable SymmetricTensor{order, dim, T <: Number, M} <: AbstractTensor{order, dim, T, M}
   data::Array{T, M}
end

immutable Tensor{order, dim, T <: Number, M} <: AbstractTensor{order, dim, T, M}
   data::Array{T, M}
end

immutable Vec{dim, T <: Number} <: AbstractTensor{1, dim, T, 1}
   data::Vector{T}
end

###############
# Typealiases #
###############
typealias AllTensors{dim, T} Union{SymmetricTensor{2, dim, T, 1}, Tensor{2, dim, T, 1},
                                   SymmetricTensor{4, dim, T, 2}, Tensor{4, dim, T, 2},
                                   Vec{dim, T}}

typealias SecondOrderTensor{dim, T} Union{SymmetricTensor{2, dim, T, 1}, Tensor{2, dim, T, 1}}
typealias FourthOrderTensor{dim, T} Union{SymmetricTensor{4, dim, T, 2}, Tensor{4, dim, T, 2}}

typealias SymmetricTensors{dim, T} Union{SymmetricTensor{2, dim, T, 1}, SymmetricTensor{4, dim, T, 2}}
typealias Tensors{dim, T} Union{Tensor{2, dim, T, 1}, Tensor{4, dim, T, 2},
                                   Vec{dim, T}}

##############################
# Utility/Accessor Functions #
##############################

get_data(t::AbstractTensor) = t.data

@inline function n_independent_components(dim, issym)
    dim == 1 && return 1
    if issym
        dim == 2 && return 3
        dim == 3 && return 6
    else
        dim == 2 && return 4
        dim == 3 && return 9
    end
    return -1
end


get_lower_order_tensor{dim, T}(S::Type{SymmetricTensor{2, dim, T, 1}}) = S
get_lower_order_tensor{dim, T}(S::Type{Tensor{2, dim, T, 1}}) = S
get_lower_order_tensor{dim, T}(::Type{SymmetricTensor{4, dim, T, 2}}) = SymmetricTensor{2, dim, T, 1}
get_lower_order_tensor{dim, T}(::Type{Tensor{4, dim, T, 2}}) = Tensor{2, dim, T, 1}


############################
# Abstract Array interface #
############################
Base.linearindexing(::SymmetricTensor) = Base.LinearSlow()
Base.linearindexing(::Tensors) = Base.LinearSlow()

# Size #
########
Base.size(::Vec{1}) = (1,)
Base.size(::Vec{2}) = (2,)
Base.size(::Vec{3}) = (3,)

Base.size(::SecondOrderTensor{1}) = (1, 1)
Base.size(::SecondOrderTensor{2}) = (2, 2)
Base.size(::SecondOrderTensor{3}) = (3, 3)

Base.size(::FourthOrderTensor{1}) = (1, 1, 1, 1)
Base.size(::FourthOrderTensor{2}) = (2, 2, 2, 2)
Base.size(::FourthOrderTensor{3}) = (3, 3, 3, 3)

Base.similar(t::AbstractTensor) = typeof(t)(similar(get_data(t)))
Base.fill!(t::AbstractTensor, v) = (fill!(get_data(t), v); return t)

# Indexing #
############
@inline function get_index_from_symbol(sym::Symbol)
    if sym == :x; i = 1;
    elseif sym == :y; i = 2;
    elseif sym == :z; i = 3;
    else
        return 0 # This will bound serror later
    end
    return i
end

@inline function Base.checkbounds{dim}(t::Vec{dim}, i::Int)
    (i <= 0 || i > dim ) && throw(BoundsError(t, (i,)))
end

@inline function Base.checkbounds{dim}(t::SecondOrderTensor{dim}, i::Int, j::Int)
    (i <= 0 || i > dim || j <= 0 || j > dim) && throw(BoundsError(t, (i, j)))
end

@inline function Base.checkbounds{dim}(t::FourthOrderTensor{dim}, i::Int, j::Int, k::Int, l::Int)
    (i <= 0 || i > dim || j <= 0 || j > dim || k <= 0 || k > dim || l <= 0 || l > dim) && throw(BoundsError(t, (i, j, k,l)))
end


@inline function compute_index{dim, T, M}(::Type{SymmetricTensor{2, dim, T, M}}, i::Int, j::Int)
    if i < j
        i, j  = j,i
    end
    skipped_indicies= div((j-1) * j, 2)
    return dim*(j-1) + i - skipped_indicies
end

@inline function compute_index{dim, T, M}(::Type{Tensor{2, dim, T, M}}, i::Int, j::Int)
    return dim*(j-1) + i
end

# getindex #
############
@inline function Base.getindex(S::Vec, i::Int)
    checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

@inline function Base.getindex(S::Vec, si::Symbol)
    i = get_index_from_symbol(si)
    checkbounds(S, i)
    @inbounds v = get_data(S)[i]
    return v
end

@inline function Base.getindex(S::SecondOrderTensor, i::Int, j::Int)
    checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(typeof(S), i, j)]
    return v
end

@inline function Base.getindex(S::SecondOrderTensor, si::Symbol, sj::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    checkbounds(S, i, j)
    @inbounds v = get_data(S)[compute_index(typeof(S), i, j)]
    return v
end

@inline function Base.getindex(S::FourthOrderTensor, i::Int, j::Int, k::Int, l::Int)
    checkbounds(S, i, j, k, l)
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    @inbounds v = get_data(S)[I, J]
    return v
end

@inline function Base.getindex(S::FourthOrderTensor, si::Symbol, sj::Symbol, sk::Symbol, sl::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    k = get_index_from_symbol(sk)
    l = get_index_from_symbol(sl)
    checkbounds(S, i, j, k, l)
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    @inbounds v = get_data(S)[I, J]
    return v
end

# setindex! #
#############
@inline function Base.setindex!(S::Vec, v, i::Int)
    checkbounds(S, i)
    @inbounds get_data(S)[i] = v
    return v
end

@inline function Base.setindex!(S::Vec, v, si::Symbol)
    i = get_index_from_symbol(si)
    checkbounds(S, i)
    @inbounds get_data(S)[i] = v
    return v
end

@inline function Base.setindex!(S::SecondOrderTensor, v, i::Int, j::Int)
    checkbounds(S, i, j)
    @inbounds get_data(S)[compute_index(typeof(S), i, j)] = v
    return v
end

@inline function Base.setindex!(S::SecondOrderTensor, v, si::Symbol, sj::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    checkbounds(S, i, j)
    @inbounds get_data(S)[compute_index(typeof(S), i, j)] = v
    return v
end

@inline function Base.setindex!(S::FourthOrderTensor, v, i::Int, j::Int, k::Int, l::Int)
    checkbounds(S, i, j, k, l)
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    @inbounds get_data(S)[I, J] = v
    return v
end

@inline function Base.setindex!(S::FourthOrderTensor, v, si::Symbol, sj::Symbol, sk::Symbol, sl::Symbol)
    i = get_index_from_symbol(si)
    j = get_index_from_symbol(sj)
    k = get_index_from_symbol(sk)
    l = get_index_from_symbol(sl)
    checkbounds(S, i, j, k, l)
    I = compute_index(get_lower_order_tensor(typeof(S)), i, j)
    J = compute_index(get_lower_order_tensor(typeof(S)), k, l)
    @inbounds get_data(S)[I, J] = v
    return v
end


################
# Constructors #
################

get_base{order, dim}(::Type{Vec{order, dim}}) = Vec
get_base{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor
get_base{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor

# Internal constructors #
#########################

function Vec{dim, T}(data::Vector{T}, ::Type{Val{dim}})
    length(data) == dim || throw(InternalError())
    Vec{dim, T}(data)
end

function SymmetricTensor{dim, T}(data::Vector{T}, ::Type{Val{dim}})
    length(data) == n_independent_components(dim, true) || throw(InternalError())
    SymmetricTensor{2, dim, T, 1}(data)
end

function SymmetricTensor{T, dim}(data::Matrix{T}, ::Type{Val{dim}})
    n = n_independent_components(dim, true)
    size(data) == (n,n)  || throw(InternalError())
    SymmetricTensor{4, dim, T, 2}(data)
end


function Tensor{T, dim}(data::Vector{T}, ::Type{Val{dim}})
    length(data) == n_independent_components(dim, false) || throw(InternalError())
    Tensor{2, dim, T, 1}(data)
end

function Tensor{T, dim}(data::Matrix{T}, ::Type{Val{dim}})
    n = n_independent_components(dim, false)
    size(data) == (n,n)  || throw(InternalError())
    Tensor{4, dim, T, 2}(data)
end

#############
# Promotion #
#############

function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{Vec{dim, A}},
                                                           ::Type{Vec{dim, B}})
    Vec{dim, promote_type(A, B)}
end


function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{SymmetricTensor{2, dim, A, 1}},
                                                           ::Type{SymmetricTensor{2, dim, B, 1}})
    SymmetricTensor{2, dim, promote_type(A, B), 1}
end

function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{SymmetricTensor{4, dim, A, 2}},
                                                           ::Type{SymmetricTensor{4, dim, B, 2}})
    SymmetricTensor{4, dim, promote_type(A, B), 2}
end

function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{Tensor{2, dim, A, 1}},
                                                           ::Type{Tensor{2, dim, B, 1}})
    Tensor{2, dim, promote_type(A, B), 1}
end

function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{Tensor{4, dim, A, 2}},
                                                           ::Type{Tensor{4, dim, B, 2}})
    Tensor{4, dim, promote_type(A, B), 2}
end

# SymmetricTensor and Tensor promotes to Tensor
#function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{Tensor{2, dim, A, 1}},
#                                                           ::Type{SymmetricTensor{2, dim, B, 1}})
#    Tensor{2, dim, promote_type(A, B), 1}
#end
#
#function Base.promote_rule{dim , A <: Number, B <: Number}(T1::Type{SymmetricTensor{2, dim, A, 1}},
#                                                           T2::Type{Tensor{2, dim, B, 1}})
#    Tensor{2, dim, promote_type(A, B), 1}
#end
#
#
#function Base.promote_rule{dim , A <: Number, B <: Number}(::Type{Tensor{4, dim, A, 1}},
#                                                           ::Type{SymmetricTensor{4, dim, B, 1}})
#    Tensor{4, dim, promote_type(A, B), 2}
#end
#
#function Base.promote_rule{dim , A <: Number, B <: Number}(T1::Type{SymmetricTensor{4, dim, A, 2}},
#                                                           T2::Type{Tensor{4, dim, B, 2}})
#    Tensor{4, dim, promote_type(A, B), 2}
#end



##############
# Conversion #
##############

#@gen_code function Base.convert{order, dim, T1, T2}(::Type{SymmetricTensor, order, dim , T1}, t::Tensor{order, dim, T2})
#    @code :(S = zeros(SymmetricTensor{order, dim , T1}))
#    @code :(data = get_data(S))
#    k = 1
#    for i in 1:dim, j in 1:dim
#        if i == j
#            @code :(@inbounds data[$k] = mat[$i,$j])
#        else
#            @code :(@inbounds data[$k] = 0.5 * (mat[$i,$j] + mat[$j,$i]))
#        end
#        k += 1
#    end
#    @code :(return S)
#end
#
#end
#
#Base.convert{order, dim, T}(Type{SymmetricTensor, order, dim}, t::Tensor{order, dim, T}) =
#    convert(Type{SymmetricTensor, order, dim, T}, t)

# Predicates #
##############

Base.isequal{dim}(a::SymmetricTensor{dim}, b::SymmetricTensor{dim}) = isequal(get_data(a), get_data(b))
Base.isequal{dim}(a::Tensor{dim}, b::Tensor{dim}) = isequal(get_data(a), get_data(b))
Base.(:(==)){dim}(a::SymmetricTensor{dim}, b::SymmetricTensor{dim}) = isequal(get_data(a), get_data(b))
Base.(:(==)){dim}(a::Tensor{dim}, b::Tensor{dim}) = isequal(get_data(a), get_data(b))


# copy / copy! #
################
function Base.copy!{order, dim}(S1::SymmetricTensor{order, dim}, S2::SymmetricTensor{order, dim})
    copy!(get_data(S1), get_data(S2))
    return S1
end

function Base.copy!{order, dim}(S1::Tensor{order, dim}, S2::Tensor{order, dim})
    copy!(get_data(S1), get_data(S2))
    return S1
end

function Base.copy{order, dim}(S2::SymmetricTensor{order, dim})
    S1 = similar(S2)
    copy!(get_data(S1), get_data(S2))
    return S1
end

function Base.copy{order, dim}(S2::Tensor{order, dim})
    S1 = similar(S)
    copy!(get_data(S1), get_data(S2))
    return S1
end


###############
# Simple Math #
###############

function Base.(:*){dim}(n::Number, t::AllTensors{dim})
    get_base(typeof(t))(n * get_data(t), Val{dim})
end

Base.(:*){dim}(t::AllTensors{dim}, n::Number) = n * t

function Base.(:/){dim}(t::AllTensors{dim}, n::Number)
    get_base(typeof(t))(get_data(t) / n, Val{dim})
end

# Minus
function Base.(:-){dim}(t1::SymmetricTensor, t2::SymmetricTensor{dim})
    get_base(typeof(t1))(get_data(t1) - get_data(t2), Val{dim})
end

function Base.(:-){dim}(t1::Tensors, t2::Tensors{dim})
    get_base(typeof(t1))(get_data(t1) - get_data(t2), Val{dim})
end

function Base.(:-){dim}(t::AllTensors{dim})
    get_base(typeof(t))( - get_data(t), Val{dim})
end

# Plus
function Base.(:+){dim}(t1::SymmetricTensor, t2::SymmetricTensor{dim})
    get_base(typeof(t1))(get_data(t1) + get_data(t2), Val{dim})
end

function Base.(:+){dim}(t1::Tensors, t2::Tensors{dim})
    get_base(typeof(t1))(get_data(t1) + get_data(t2), Val{dim})
end

function Base.(:+){dim}(t::AllTensors{dim})
    get_base(typeof(t))(get_data(t), Val{dim})
end


###################
# Zero, one, rand #
###################
for order in (2, 4)
    for (issym, T) in ((true, SymmetricTensor), (false, Tensor))
        for (f, f2) in ((:zeros, :zero), (:rand, :rand))
            if order == 2
                data = quote data = $f(T, n) end
            else
                data = quote data = $f(T, n, n) end
            end
            @eval begin
                function Base.$f2{dim, T}(S::Type{$T{$order, dim, T}})
                    n = n_independent_components(dim, $issym)
                    $data
                    return $T(data, Val{dim})
                end

                function Base.$f2{dim}(S::Type{$T{$order, dim}})
                    Base.$f2($T{$order, dim, Float64})
                end

                function Base.$f2{dim, T, M}(S::Type{$T{$order, dim, T, M}})
                    Base.$f2($T{$order, dim, T})
                end

                function Base.$f2(S::$T)
                    Base.$f2(typeof(S))
                end
            end
        end
    end
end

function Base.one{dim, T}(ST::Type{SecondOrderTensor{dim, T}})
    n = n_independent_components(ST)
    S = ST(zeros(n,n), Val{dim})
    @inbounds for i in 1:dim
        S[i,i] = one(T)
    end
    return S
end

function Base.one{dim, T}(S::SecondOrderTensor{dim, T})
    S_new = S
    @inbounds for i in 1:dim
        S_new[i,i] = one(T)
    end
    return S_new
end

function Base.one{dim, T}(S::FourthOrderTensor{dim, T})
    S_new = zero(S)
    @inbounds for i in 1:dim
        S_new[i,i,i,i] = one(T)
    end
    return S_new
end


Base.zero(T::Vec) = zero(typeof(T))
Base.one(T::Vec) = one(typeof(T))
Base.rand(T::Vec) = rand(typeof(T))

Base.zero{dim, T}(::Type{Vec{dim,T}}) = Vec(zeros(T, dim), Val{dim})
Base.one{dim, T}(::Type{Vec{dim,T}}) = Vec(ones(T, dim), Val{dim})
Base.rand{dim, T}(::Type{Vec{dim,T}}) = Vec(rand(T, dim), Val{dim})


Base.zero{dim}(::Type{Vec{dim}}) = Vec(zeros(Float64, dim), Val{dim})
Base.one{dim}(::Type{Vec{dim}}) = Vec(ones(Float64, dim), Val{dim})
Base.rand{dim}(::Type{Vec{dim}}) = Vec(rand(Float64, dim), Val{dim})

include("utilities.jl")
include("symmetric_ops.jl")
include("tensor_ops.jl")
include("data_functions.jl")


end # module
