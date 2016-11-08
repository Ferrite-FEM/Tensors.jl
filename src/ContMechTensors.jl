__precompile__()

module ContMechTensors

immutable InternalError <: Exception end

export AbstractTensor, SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

export otimes, ⊗, ⊡, dcontract, dev, vol, symmetric, skew, minorsymmetric, majorsymmetric
export minortranspose, majortranspose, isminorsymmetric, ismajorsymmetric
export setindex, tdot, dotdot

@deprecate extract_components(tensor) Array(tensor)

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
typealias Tensors{dim, T} Union{Tensor{2, dim, T}, Tensor{4, dim, T}, Vec{dim, T}}

include("utilities.jl")
include("tuple_utils.jl")
include("tuple_linalg.jl")
include("symmetric_tuple_linalg.jl")

include("indexing.jl")
include("promotion_conversion.jl")
include("tensor_ops.jl")
include("tensor_ops_errors.jl")
include("symmetric_ops.jl")


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


########
# Size #
########

Base.size{dim}(::Vec{dim}) = (dim,)
Base.size{dim}(::SecondOrderTensor{dim}) = (dim, dim)
Base.size{dim}(::FourthOrderTensor{dim}) = (dim, dim, dim, dim)

Base.similar(t::AbstractTensor) = typeof(t)(get_data(t))

# Ambiguity fix
Base.fill(t::AbstractTensor, v::Integer)  = typeof(t)(const_tuple(typeof(get_data(t)), v))
Base.fill(t::AbstractTensor, v::Number) = typeof(t)(const_tuple(typeof(get_data(t)), v))


#########################
# Internal constructors #
#########################

@inline function (Tt::Union{Type{Tensor{order, dim, T, M}}, Type{SymmetricTensor{order, dim, T, M}}}){order, dim, T, M}(data)
    get_base(Tt)(data)
end

@inline function (Tt::Type{Vec{dim}}){dim}(data)
    Tensor{1, dim}(data)
end

# These are some kinda ugly stuff to create different type of constructors.
@generated function (Tt::Type{Tensor{order, dim}}){order, dim}(data::Union{AbstractArray, Tuple})
    # Check for valid orders
    n = n_components(Tensor{order,dim})
    if !(order in (1,2,4))
        throw(ArgumentError("Tensor only supported for order 1, 2, 4"))
    end
    return quote
        if length(data) != $n
            throw(ArgumentError("Wrong number of tuple elements, expected $($n), got $(length(data))"))
        end
        Tensor{order, dim, eltype(data), $n}(to_tuple(NTuple{$n}, data))
    end
end

# These are some kinda ugly stuff to create different type of constructors.
@generated function (Tt::Type{SymmetricTensor{order, dim}}){order, dim}(data::Union{AbstractArray, Tuple})
    n = n_components(Tensor{order,dim})
    m = n_components(SymmetricTensor{order,dim})
    if !(order in (2,4))
        throw(ArgumentError("SymmetricTensor only supported for order 2, 4"))
    end
    return quote
        if length(data) != $n && length(data) != $m
            throw(ArgumentError("Wrong number of tuple elements, expected $($n) or $($m), got $(length(data))"))
        end
        if length(data) == $m
            return SymmetricTensor{order, dim, eltype(data), $m}(to_tuple(NTuple{$m}, data))
        end
        S = Tensor{order, dim, eltype(data), $n}(to_tuple(NTuple{$n}, data))
        return convert(SymmetricTensor{order, dim}, S)
    end
end


@generated function (Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}){order, dim}(f::Function)
    # Check for valid orders
    if !(order in (1,2,4))
        throw(ArgumentError("Only tensors of order 1, 2, 4 supported"))
    end
 
    # Storage format is of rank 1 for vectors and order / 2 for other tensors
    if order == 1 && Tt <: SymmetricTensor
        throw(ArgumentError("SymmetricTensor only supported for order 2, 4"))
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
    
    return :(get_main_type(Tt){order, dim}($exp))
end


###############
# Simple Math #
###############

Base.:*(t::AllTensors, n::Number) = n * t

for TensorType in (SymmetricTensor, Tensor)
    @eval begin
        @generated function Base.:*{order,dim, T, M}(n::Number, t::$(TensorType){order, dim, T, M})
            exp = tensor_create_elementwise(get_base(t), (k) -> :(n * t.data[$k]))
            Tv = typeof(zero(n) * zero(T))
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $($TensorType){order, dim, $Tv, M}($exp)
                return t
            end
        end

        @generated function Base.:/{order,dim, T, M}(t::$(TensorType){order, dim, T, M}, n::Number)
            exp = tensor_create_elementwise(get_base(t), (k) -> :(t.data[$k] / n))
            Tv = typeof(zero(n) / zero(T))
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $($TensorType){order, dim, $Tv, M}($exp)
                return t
            end
        end

        @generated function Base.:-{order,dim, T, M}(t::$(TensorType){order, dim, T, M})
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


##########################
# Zero, one, rand, diagm #
##########################

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

        @generated function Base.diagm{order, dim, T}(Tt::Type{$(TensorType){order, dim}}, v::AbstractVector{T})
            N = n_components($(TensorType){order, dim})
            if order == 1
                f = (i) -> :(v[$i])
            elseif order == 2
                f = (i,j) -> i == j ? :(v[$i]) : :($(zero(T)))
            elseif order == 4
                f = (i,j,k,l) -> i == k && j == l ? :(v[$i]) : :($(zero(T)))
            end
            exp = tensor_create(get_type(Tt),f)
            return quote
                $(Expr(:meta, :inline))
                @inbounds t = $exp
                $($TensorType){order, dim, T, $N}(t)
            end
        end

        @generated function Base.diagm{order, dim, T}(Tt::Type{$(TensorType){order, dim}}, v::T)
            N = n_components($(TensorType){order, dim})
            if order == 1
                f = (i) -> :(v)
            elseif order == 2
                f = (i,j) -> i == j ? :(v) : :($(zero(T)))
            elseif order == 4
                f = (i,j,k,l) -> i == k && j == l ? :(v) : :($(zero(T)))
            end
            exp = tensor_create(get_type(Tt),f)
            return quote
                $(Expr(:meta, :inline))
                $($TensorType){order, dim, T, $N}($exp)
            end
        end

        function Base.one{order, dim, T}(Tt::Type{$(TensorType){order, dim, T}})
            Base.diagm($(TensorType){order, dim}, one(T))
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
