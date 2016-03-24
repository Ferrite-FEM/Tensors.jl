#__precompile__()

module ContMechTensors


if VERSION <= v"0.5.0-dev"
    macro boundscheck(exp)
        esc(exp)
    end
end

macro L_str(text)
    text
end

immutable InternalError <: Exception end

export AbstractTensor, SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

export otimes, otimes_unsym, ⊗, ⊡, dcontract, dev, dev!
export extract_components, load_components!, symmetrize, symmetrize!
export setindex, store!, tdot

#########
# Types #
#########
abstract AbstractTensor{order, dim, T <: Real} <: AbstractArray{T, order}


immutable SymmetricTensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

immutable Tensor{order, dim, T <: Real, M} <: AbstractTensor{order, dim, T}
   data::NTuple{M, T}
end

###############
# Typealiases #
###############


typealias Vec{dim, T, M} Tensor{1, dim, T, dim}

typealias AllTensors{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T},
                                   SymmetricTensor{4, dim, T}, Tensor{4, dim, T},
                                   Vec{dim, T}}


typealias SecondOrderTensor{dim, T} Union{SymmetricTensor{2, dim, T}, Tensor{2, dim, T}}
typealias FourthOrderTensor{dim, T} Union{SymmetricTensor{4, dim, T}, Tensor{4, dim, T}}
typealias SymmetricTensors{dim, T} Union{SymmetricTensor{2, dim, T}, SymmetricTensor{4, dim, T}}
typealias Tensors{dim, T} Union{Tensor{2, dim, T}, Tensor{4, dim, T},
                                   Vec{dim, T}}

include("utilities.jl")
include("tuple_utils.jl")
include("tuple_linalg.jl")
include("symmetric_tuple_linalg.jl")

include("indexing.jl")
include("promotion_conversion.jl")
include("tensor_ops.jl")
include("symmetric_ops.jl")
include("data_functions.jl")



##############################
# Utility/Accessor Functions #
##############################

get_data(t::AbstractTensor) = t.data

function n_independent_components(dim, issym)
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

n_components{dim}(::Type{SymmetricTensor{2, dim}}) = dim*dim - div((dim-1)*dim, 2)
function n_components{dim}(::Type{SymmetricTensor{4, dim}})
    n = n_components(SymmetricTensor{2, dim})
    return n*n
end

@inline n_components{order, dim}(::Type{Tensor{order, dim}}) = dim^order

@inline is_always_sym{dim, T}(::Type{Tensor{dim, T}}) = false
@inline is_always_sym{dim, T}(::Type{SymmetricTensor{dim, T}}) = true

@inline get_main_type{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor
@inline get_main_type{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor
@inline get_main_type{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor
@inline get_main_type{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor
@inline get_main_type{order, dim}(::Type{SymmetricTensor{order, dim}}) = SymmetricTensor
@inline get_main_type{order, dim}(::Type{Tensor{order, dim}}) = Tensor

@inline get_base{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor{order, dim}
@inline get_base{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor{order, dim}

@inline get_lower_order_tensor{dim, T, M}(S::Type{SymmetricTensor{2, dim, T, M}}) =  SymmetricTensor{2, dim}
@inline get_lower_order_tensor{dim, T, M}(S::Type{Tensor{2, dim, T, M}}) = Tensor{2, dim}
@inline get_lower_order_tensor{dim, T, M}(::Type{SymmetricTensor{4, dim, T, M}}) = SymmetricTensor{2, dim}
@inline get_lower_order_tensor{dim, T, M}(::Type{Tensor{4, dim, T, M}}) = Tensor{2, dim}


############################
# Abstract Array interface #
############################

Base.linearindexing{T <: SymmetricTensor}(::Type{T}) = Base.LinearSlow()
Base.linearindexing{T <: Tensor}(::Type{T}) = Base.LinearFast()

get_type{X}(::Type{Type{X}}) = X

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

Base.similar(t::AbstractTensor) = typeof(t)(get_data(t))

# Ambiguity fix
Base.fill(t::AbstractTensor, v::Integer)  = typeof(t)(const_tuple(typeof(get_data(t)), v))
Base.fill(t::AbstractTensor, v::Number) = typeof(t)(const_tuple(typeof(get_data(t)), v))


# Internal constructors #
#########################

function call{order, dim, T, M}(Tt::Union{Type{Tensor{order, dim, T, M}}, Type{SymmetricTensor{order, dim, T, M}}},
                                       data)
    get_base(Tt)(data)
end


## These are some kinda ugly stuff to create different type of constructors.
@gen_code function call{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}},
                                          data)
    # Check for valid orders
    if !(order in (1,2,4))
        @code (throw(ArgumentError("Only tensors of order 1, 2, 4 supported")))
    else
        # Storage format is of rank 1 for vectors and order / 2 for other tensors
        if order == 1
            @code(:(Tt <: SymmetricTensor && throw(ArgumentError("SymmetricTensor only supported for order 2, 4"))))
        end

        n = n_components(get_type(Tt))

        # Validate that the input array has the correct number of elements.
        @code :(length(data) == $n || throw(ArgumentError("wrong number of tuple elements, expected $($n), got $(length(data))")))
        @code :(get_main_type(Tt){order, dim, eltype(data), $n}(to_tuple(NTuple{$n}, data)))
    end
end

@gen_code function call{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}},
                                          f::Function)
    # Check for valid orders
    if !(order in (1,2,4))
        @code (throw(ArgumentError("Only tensors of order 1, 2, 4 supported")))
    else
        # Storage format is of rank 1 for vectors and order / 2 for other tensors
        if order == 1
            @code(:(Tt <: SymmetricTensor && throw(ArgumentError("SymmetricTensor only supported for order 2, 4"))))
        end

        n = n_components(get_type(Tt))

        # Validate that the input array has the correct number of elements.
        if order == 1
            exp = tensor_create(get_main_type(get_type(Tt)){order, dim}, (i) -> :(f($i)))
        elseif order == 2
            exp = tensor_create(get_main_type(get_type(Tt)){order, dim}, (i,j) -> :(f($i, $j)))
        elseif order == 4
            exp = tensor_create(get_main_type(get_type(Tt)){order, dim}, (i,j,k,l) -> :(f($i, $j, $k, $l)))
        end
        @code :(get_main_type(Tt){order, dim}($exp))
    end
end

###############
# Simple Math #
###############
Base.(:*)(t::AllTensors, n::Number) = n * t

for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @generated function Base.(:*){order,dim, T, M}(n::Number, t::$(TensorType){order, dim, T, M})
            exp = tensor_create_elementwise(get_base(t), (k) -> :(n * t.data[$k]))
            Tv = typeof(zero(n) * zero(T))
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $($TensorType){order, dim, $Tv, M}($exp)
                return t
            end
        end

        @generated function Base.(:/){order,dim, T, M}(t::$(TensorType){order, dim, T, M}, n::Number)
            exp = tensor_create_elementwise(get_base(t), (k) -> :(t.data[$k] / n))
            Tv = typeof(zero(n) / zero(T))
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $($TensorType){order, dim, $Tv, M}($exp)
                return t
            end
        end

        @generated function Base.(:-){order,dim, T, M}(t::$(TensorType){order, dim, T, M})
            exp = tensor_create_elementwise(get_base(t), (k) -> :(-t.data[$k]))
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $($TensorType){order, dim, T, M}($exp)
                return t
            end
        end
    end
end

for (op, fun) in ((:-, (k) -> :(t1.data[$k] - t2.data[$k])),
                  (:+, (k) -> :(t1.data[$k] + t2.data[$k])),
                  (:.*, (k) -> :(t1.data[$k] * t2.data[$k])),
                  (:./, (k) -> :(t1.data[$k] / t2.data[$k])))
    for TensorType in (SymmetricTensor, Tensor)
        @eval begin
            @generated function Base.$op{order, dim, T1, T2, M}(t1::$(TensorType){order, dim, T1, M}, t2::$(TensorType){order, dim, T2, M})
                exp = tensor_create_elementwise(get_base(t1), $fun)
                Tv = typeof( $op(zero(T1), zero(T2)))
                return quote
                    $(Expr(:meta, :inline))
                    @inbounds t = $($TensorType){order, dim, $Tv, M}($exp)
                    return t
                end
            end
        end
    end

    @eval begin
        Base.$op{order, dim}(t1::AbstractTensor{order, dim}, t2::AbstractTensor{order, dim}) = Base.$op(promote(t1, t2)...)
    end
end

###################
# Zero, one, rand #
###################

for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @generated function Base.rand{order, dim, T}(Tt::Type{$(TensorType){order, dim, T}})
            N = n_components($(TensorType){order, dim})
            exp = tensor_create_no_arg(get_type(Tt), () -> :(rand(T)))
            return quote
                $(Expr(:meta, :inline))
                $($TensorType){order, dim, T, $N}($exp)
            end
        end

        @generated function Base.zero{order, dim, T}(Tt::Type{$(TensorType){order, dim, T}})
            N = n_components($(TensorType){order, dim})
            exp = tensor_create_no_arg(get_type(Tt), () -> :(zero(T)))
            return quote
                $(Expr(:meta, :inline))
                $($TensorType){order, dim, T, $N}($exp)
            end
        end

        @generated function Base.one{order, dim, T}(Tt::Type{$(TensorType){order, dim, T}})
            N = n_components($(TensorType){order, dim})
            if order == 1
                f = (i) -> :($(one(T)))
            elseif order == 2
                f = (i,j) -> i == j ? :($(one(T))) : :($(zero(T)))
            elseif order == 4
                f = (i,j,k,l) -> i == k && j == l ? :($(one(T))) : :($(zero(T)))
            end
            exp = tensor_create(get_type(Tt),f)
            return quote
                $(Expr(:meta, :inline))
                $($TensorType){order, dim, T, $N}($exp)
            end
        end
    end
end


for f in (:zero, :rand, :one)
    for TensorType in (SymmetricTensor, Tensor)
        @eval begin
            @inline function Base.$f{order, dim}(Tt::Type{$(TensorType){order, dim}})
                $f($(TensorType){order, dim, Float64})
            end

            @inline function Base.$f{order, dim, T, M}(Tt::Type{$(TensorType){order, dim, T, M}})
                $f($(TensorType){order, dim, T})
            end
        end
    end

    @eval begin
     # Special for Vec
        @inline function Base.$f{dim}(Tt::Type{Vec{dim}})
            $f(Vec{dim, Float64})
        end

        function Base.$f(t::AllTensors)
            $f(typeof(t))
        end
    end
end

end # module
