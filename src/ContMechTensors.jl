#__precompile__()

module ContMechTensors


if VERSION <= v"0.5.0-dev"
    macro boundscheck(exp)
        esc(exp)
    end
end


immutable InternalError <: Exception end

export SymmetricTensor, Tensor, Vec, FourthOrderTensor, SecondOrderTensor

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

n_components{order, dim}(::Type{Tensor{order, dim}}) = dim^order

is_always_sym{dim, T}(::Type{Tensor{dim, T}}) = false
is_always_sym{dim, T}(::Type{SymmetricTensor{dim, T}}) = true


get_main_type{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}) = SymmetricTensor
get_main_type{order, dim, T}(::Type{Tensor{order, dim, T}}) = Tensor
get_main_type{order, dim}(::Type{SymmetricTensor{order, dim}}) = SymmetricTensor
get_main_type{order, dim}(::Type{Tensor{order, dim}}) = Tensor

#
get_base{order, dim, T, M}(::Type{SymmetricTensor{order, dim, T, M}}) = SymmetricTensor{order, dim}
get_base{order, dim, T, M}(::Type{Tensor{order, dim, T, M}}) = Tensor{order, dim}
#
get_lower_order_tensor{dim, T, M}(S::Type{SymmetricTensor{2, dim, T, M}}) =  SymmetricTensor{2, dim}
get_lower_order_tensor{dim, T, M}(S::Type{Tensor{2, dim, T, M}}) = Tensor{2, dim}
get_lower_order_tensor{dim, T, M}(::Type{SymmetricTensor{4, dim, T, M}}) = SymmetricTensor{2, dim}
get_lower_order_tensor{dim, T, M}(::Type{Tensor{4, dim, T, M}}) = Tensor{2, dim}


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



###############
# Simple Math #
###############

function Base.(:*)(n::Number, t::AllTensors)
     typeof(t)(scale_tuple(get_data(t), n))
end

Base.(:*)(t::AllTensors, n::Number) = n * t

function Base.(:/)(t::AllTensors, n::Number)
    typeof(t)(div_tuple_by_scalar(get_data(t), n))
end

function Base.(:-)(t::AllTensors)
    typeof(t)(minus_tuple(get_data(t)))
end
                            # The smilieys...
for (f, tuple_f) in zip((:(:-), :(:+), :(:.*), :(:./)), (:subtract_tuples, :add_tuples, :scalar_mul_tuples, :scalar_div_tuples))
    @eval begin
        function Base.($f){dim}(t1::AllTensors{dim}, t2::AllTensors{dim})
            a, b = promote(t1, t2)
            typeof(a)($tuple_f(get_data(a), get_data(b)))
        end
    end
end


###################
# Zero, one, rand #
###################

@gen_code function Base.rand{order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}})
    n = n_components(get_main_type(get_type(Tt)){order, dim})
    @code :(get_main_type(Tt){order, dim}(rand_tuple(NTuple{$n, T})))
end

@gen_code function Base.zero{order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}})
    n = n_components(get_main_type(get_type(Tt)){order, dim})
    @code :(get_main_type(Tt){order, dim}(zero_tuple(NTuple{$n, T})))
end

@gen_code function Base.one{order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}})
    n = n_components(get_main_type(get_type(Tt)){order, dim})
    @code :(get_main_type(Tt){order, dim}(eye_tuple(NTuple{$n, T})))
end

@gen_code function Base.one{order, dim, T}(Tt::Type{Tensor{order, dim, T}})
    n = n_components(get_main_type(get_type(Tt)){order, dim})
    @code :(get_main_type(Tt){order, dim}(eye_tuple(NTuple{$n, T})))
end

@gen_code function Base.one{order, dim, T}(Tt::Type{SymmetricTensor{order, dim, T}})
    n = n_components(get_main_type(get_type(Tt)){order, dim})
    @code :(get_main_type(Tt){order, dim}(sym_eye_tuple(NTuple{$n, T})))
end

function Base.one{dim, T}(Tt::Type{Tensor{1, dim, T}})
    get_main_type(Tt){1, dim}(const_tuple(NTuple{dim, T}, one(T)))
end

for f in (:zero, :rand, :one)
    @eval begin
        function Base.$f{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}})
            $f(get_main_type(Tt){order, dim, Float64})
        end

        function Base.$f{order, dim, T, M}(Tt::Union{Type{Tensor{order, dim, T, M}}, Type{SymmetricTensor{order, dim, T, M}}})
            $f(get_base(Tt))
        end

        function Base.$f(t::AllTensors)
            $f(typeof(t))
        end
    end
end

end # module
