import ForwardDiff: Dual, partials, value, Tag

@static if isdefined(LinearAlgebra, :gradient)
    import LinearAlgebra.gradient
end

######################
# Extraction methods #
######################

# Extractions are supposed to unpack the value and the partials
# The partials should be put into a tensor of higher order.
# The extraction methods need to know the input type to the function
# that generated the result. The reason for this is that there is no
# difference in the output type (except the number of partials) for
# norm(v) and det(T) where v is a vector and T is a second order tensor.

####################
# Value extraction #
####################

# Scalar output -> Scalar value
"""
    function _extract_value(v::ForwardDiff.Dual)
    function _extract_value(v::AbstractTensor{<:Any,<:Any,<:Dual})
    
Extract the non-dual part of a tensor with dual entries. This 
function is useful when inserting analytical derivatives using
the [`_insert_gradient`](@ref) function
"""
@inline function _extract_value(v::Dual)
    return value(v)
end
# AbstractTensor output -> AbstractTensor gradient
@generated function _extract_value(v::AbstractTensor{<:Any,<:Any,<:Dual})
    TensorType = get_base(v)
    ex = Expr(:tuple)
    for i in 1:n_components(TensorType)
        # Can use linear indexing even for SymmetricTensor
        # when indexing the underlying tuple
        push!(ex.args, :(value(get_data(v)[$i])))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($ex)
    end
end

#######################
# Gradient extraction #
#######################

# Scalar output, Scalar input -> Scalar gradient
@inline function _extract_gradient(v::Dual, ::Number)
    return @inbounds partials(v)[1]
end
# Vec, Tensor{2/4}, SymmetricTensor{2/4} output, Scalar input -> Vec, Tensor{2/4}, SymmetricTensor{2/4} gradient
@generated function _extract_gradient(v::AbstractTensor{<:Any,<:Any,<:Dual}, ::Number)
    TensorType = get_base(v)
    ex = Expr(:tuple)
    for i in 1:n_components(TensorType)
        # Can use linear indexing even for SymmetricTensor
        # when indexing the underlying tuple
        push!(ex.args, :(partials(get_data(v)[$i])[1]))
    end
    quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($ex)
    end
end
# Scalar output, Vec input -> Vec gradient
@inline function _extract_gradient(v::Dual, ::Vec{N}) where {N}
    return Vec{N}(partials(v).values)
end
# Scalar output, SecondOrderTensor input -> SecondOrderTensor gradient
@inline function _extract_gradient(v::Dual, ::SymmetricTensor{2, dim}) where {dim}
    return SymmetricTensor{2, dim}(partials(v).values)
end
@inline function _extract_gradient(v::Dual, ::Tensor{2, dim}) where {dim}
    return Tensor{2, dim}(partials(v).values)
end

# Vec output, Vec input -> Tensor{2} gradient
@inline function _extract_gradient(v::Vec{1, <: Dual}, ::Vec{1})
    @inbounds begin
        p1 = partials(v[1])
        ∇f = Tensor{2, 1}((p1[1],))
    end
    return ∇f
end
# Vec output, Vec input -> Tensor{2} gradient
@inline function _extract_gradient(v::Vec{2, <: Dual}, ::Vec{2})
    @inbounds begin
        p1, p2 = partials(v[1]), partials(v[2])
        ∇f = Tensor{2, 2}((p1[1], p2[1], p1[2], p2[2]))
    end
    return ∇f
end
# Vec output, Vec input -> Tensor{2} gradient
@inline function _extract_gradient(v::Vec{3, <: Dual}, ::Vec{3})
    @inbounds begin
        p1, p2, p3 = partials(v[1]), partials(v[2]), partials(v[3])
        ∇f = Tensor{2, 3}((p1[1], p2[1], p3[1], p1[2], p2[2], p3[2], p1[3], p2[3], p3[3]))
    end
    return ∇f
end

# Tensor{2} output, Tensor{2} input -> Tensor{4} gradient
@inline function _extract_gradient(v::Tensor{2, 1, <: Dual}, ::Tensor{2, 1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = Tensor{4, 1}((p1[1],))
    end
    return ∇f
end
# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient(v::SymmetricTensor{2, 1, <: Dual}, ::SymmetricTensor{2, 1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = SymmetricTensor{4, 1}((p1[1],))
    end
    return ∇f
end
# Tensor{2} output, Vec input -> Tensor{3} gradient
@inline function _extract_gradient(v::Tensor{2, 1, <: Dual}, ::Vec{1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = Tensor{3, 1}((p1[1],))
    end
    return ∇f
end
# Tensor{2} output, Tensor{2} input -> Tensor{4} gradient
@inline function _extract_gradient(v::Tensor{2, 2, <: Dual}, ::Tensor{2, 2})
    @inbounds begin
        p1, p2, p3, p4 = partials(v[1,1]), partials(v[2,1]), partials(v[1,2]), partials(v[2,2])
        ∇f = Tensor{4, 2}((p1[1], p2[1], p3[1], p4[1],
                           p1[2], p2[2], p3[2], p4[2],
                           p1[3], p2[3], p3[3], p4[3],
                           p1[4], p2[4], p3[4], p4[4]))
    end
    return ∇f
end
# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient(v::SymmetricTensor{2, 2, <: Dual}, ::SymmetricTensor{2, 2})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[2,2])
        ∇f = SymmetricTensor{4, 2}((p1[1], p2[1], p3[1],
                                    p1[2], p2[2], p3[2],
                                    p1[3], p2[3], p3[3]))
    end
    return ∇f
end
# Tensor{2} output, Vec input -> Tensor{3} gradient
@inline function _extract_gradient(v::Tensor{2, 2, <: Dual}, ::Vec{2})
    @inbounds begin
        p1, p2, p3, p4 = partials(v[1,1]), partials(v[2,1]), partials(v[1,2]), partials(v[2,2])
        ∇f = Tensor{3, 2}((p1[1], p2[1], p3[1], p4[1],
                           p1[2], p2[2], p3[2], p4[2]))
    end
    return ∇f
end
# Tensor{2} output, Tensor{2} input -> Tensor{4} gradient
@inline function _extract_gradient(v::Tensor{2, 3, <: Dual}, ::Tensor{2, 3})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[3,1])
        p4, p5, p6 = partials(v[1,2]), partials(v[2,2]), partials(v[3,2])
        p7, p8, p9 = partials(v[1,3]), partials(v[2,3]), partials(v[3,3])
        ∇f = Tensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1],
                           p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2],  #    ###  #
                           p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3], p9[3],  #    # #  #
                           p1[4], p2[4], p3[4], p4[4], p5[4], p6[4], p7[4], p8[4], p9[4],  ###  ###  ###
                           p1[5], p2[5], p3[5], p4[5], p5[5], p6[5], p7[5], p8[5], p9[5],
                           p1[6], p2[6], p3[6], p4[6], p5[6], p6[6], p7[6], p8[6], p9[6],
                           p1[7], p2[7], p3[7], p4[7], p5[7], p6[7], p7[7], p8[7], p9[7],
                           p1[8], p2[8], p3[8], p4[8], p5[8], p6[8], p7[8], p8[8], p9[8],
                           p1[9], p2[9], p3[9], p4[9], p5[9], p6[9], p7[9], p8[9], p9[9]))
    end
    return ∇f
end
# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient(v::SymmetricTensor{2, 3, <: Dual}, ::SymmetricTensor{2, 3})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[3,1])
        p4, p5, p6 = partials(v[2,2]), partials(v[3,2]), partials(v[3,3])
        ∇f = SymmetricTensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1],
                                    p1[2], p2[2], p3[2], p4[2], p5[2], p6[2],
                                    p1[3], p2[3], p3[3], p4[3], p5[3], p6[3],
                                    p1[4], p2[4], p3[4], p4[4], p5[4], p6[4],
                                    p1[5], p2[5], p3[5], p4[5], p5[5], p6[5],
                                    p1[6], p2[6], p3[6], p4[6], p5[6], p6[6]))
    end
    return ∇f
end

# Tensor{2} output, Vec input -> Tensor{3} gradient
@inline function _extract_gradient(v::Tensor{2, 3, <: Dual}, ::Vec{3})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[3,1])
        p4, p5, p6 = partials(v[1,2]), partials(v[2,2]), partials(v[3,2])
        p7, p8, p9 = partials(v[1,3]), partials(v[2,3]), partials(v[3,3])
        ∇f = Tensor{3, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1],
                           p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2],  
                           p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3], p9[3]))
    end
    return ∇f
end

# for non dual variable
@inline function _extract_value(v::Any)
    return v
end
for TensorType in (Tensor, SymmetricTensor)
    @eval begin
        @inline function _extract_gradient(v::T, x::$TensorType{order, dim}) where {T<:Real, order, dim}
            zero($TensorType{order, dim, T})
        end
        @generated function _extract_gradient(v::$TensorType{order, dim, T}, ::$TensorType{order, dim}) where {T<:Real, order, dim}
            RetType = $TensorType{order+order, dim, T}
            return quote
                $(Expr(:meta, :inline))
                zero($RetType)
            end
        end
    end
end

######################
# Gradient insertion #
######################

# Insertions get the real value and derivative of a function, as well 
# a tensor of dual values that was the initial input to that function. 
# A new tensor of dual values are then created, to emulate the function
# being run with dual numbers (i.e. inserting the analytical gradient)
# As opposed to with gradient extraction, we don't have the original input 
# (scalar or tensor) to the gradient function. But we can create this based
# on the tag created in the `gradient` function. 
# Specifically, consider a function y=f(g(x)) where we want to supply the 
# derivative df/dg (at g(x)). We then have "dy/dx = df/dg dg/dx" where
# the type of product is given by the type of g:
# g is 0th order: open product ^1 
# g is 1st order: single contraction
# g is 2nd order: double contraction
# 
# ^1: Regular multiplication for scalars, but in case x and f
#     are vectors, then it is open product.
#
# Support is given for the following function configurations
# g (input)     f (output)  dfdg (derivative)
# 2nd order     0th order   2nd order
# 1st order     1st order   2nd order
# 2nd order     2nd order   4th order

# First, we define the API macro used to supply the analytical derivative
"""
    @implement_gradient(f, f_dfdx)

This macro allows specifying a function `f_dfdx` that provides an analytical 
derivative of the function `f`, and is invoked when `f` is differentiated 
using automatic differentiation based on `ForwardDiff.jl`
(e.g. when using `Tensors.jl`'s 
[`gradient`](@ref) or [`hessian`](@ref)), or one of `ForwardDiff.jl`'s API).
The function `f_dfdx` must take
the same argument as `f` and should return both the value of `f` and 
the gradient, i.e. `fval, dfdx_val = f_dfdx(x)`. The following combinations
of input and output types are supported:

| `x`                 | `f(x)`              | `dfdx`              |
|:--------------------|:--------------------|:--------------------|
| `Number`            | `Number`            | `Number`            |
| `Number`            | `Vec`               | `Vec`               |
| `Number`            | `SecondOrderTensor` | `SecondOrderTensor` |
| `Vec`               | `Number`            | `Vec`               |
| `Vec`               | `Vec`               | `Tensor{2}`         |
| `SecondOrderTensor` | `Number`            | `SecondOrderTensor` |
| `SecondOrderTensor` | `SecondOrderTensor` | `FourthOrderTensor` |

Note that if one tensor if of symmetric type, then all tensors must 
be of symmetric type

"""
macro implement_gradient(f, f_dfdx)
    return :($(esc(f))(x :: Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual}) = _propagate_gradient($(esc(f_dfdx)), x))
end
# which calls the general function _propagate_gradient that calls the specialized _insert_gradient method below
function _propagate_gradient(f_dfdx::Function, x::Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual})
    fval, dfdx_val = f_dfdx(_extract_value(x))
    return _insert_gradient(fval, dfdx_val, x)
end

# Define the _insert_gradient method
"""
    _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::ForwardDiff.Dual)
    _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::Vec{<:Any,<:ForwardDiff.Dual})
    _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::SecondOrderTensor{<:Any,<:ForwardDiff.Dual})

Allows inserting an analytical gradient for use with automatic differentiation.
Consider a composed function ``h(f(g(x)))``, where you have an efficient way to
calculate ``\\partial f/\\partial g``, but want to use automatic 
differentiation for the other functions. Then, you can make another definition 
of ``f(g)`` to dispatch on if ``g`` is a tensor with `ForwardDiff.Dual` 
entires, i.e.
```julia
function f(g::Tensor{2,dim,T}) where{dim, T<:ForwardDiff.Dual}
    gval = _extract_value(g)               # Get the non-dual tensor value
    fval = f(gval)                        # Calculate function value
    dfdg = dfdg_analytical(fval, gval)    # Calculate analytical derivative
    return _insert_gradient(fval, dfdg, g) # Return the updated dual tensor
end
```

"""
function _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::Dual{Tg}) where{Tg}
    dgdx = _extract_gradient(g, _get_original_gradient_input(g))
    dfdx = dfdg ⊗ dgdx
    return _insert_full_gradient(f, dfdx, Tg())
end

function _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::Vec{<:Any, <:Dual{Tg}}) where{Tg}
    dgdx = _extract_gradient(g, _get_original_gradient_input(g))
    dfdx = dfdg ⋅ dgdx
    return _insert_full_gradient(f, dfdx, Tg())
end

function _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::SecondOrderTensor{<:Any,<:Dual{Tg}}) where{Tg}
    dgdx = _extract_gradient(g, _get_original_gradient_input(g))
    dfdx = dfdg ⊡ dgdx
    return _insert_full_gradient(f, dfdx, Tg())
end

# Define helper function to figure out original input to gradient function
_get_original_gradient_input(::Dual{Tag{Tf,Tv}}) where{Tf,Tv} = zero(Tv)
_get_original_gradient_input(::AbstractTensor{<:Any,<:Any,<:Dual{Tag{Tf,Tv}}}) where{Tf,Tv} = zero(Tv)

# Define helper function to insert_the_full_gradient calculated in _insert_gradient
_insert_full_gradient(f::Number, dfdx::Number, ::Tg) where{Tg} = Dual{Tg}(f, dfdx)
_insert_full_gradient(f::Number, dfdx::AbstractTensor, ::Tg) where{Tg} = Dual{Tg}(f, get_data(dfdx))

function _insert_full_gradient(f::TT, dfdx::TT, ::Tg) where{TT<:AbstractTensor,Tg}
    fdata = get_data(f)
    diffdata = get_data(dfdx)
    TTb = get_base(TT)
    @inbounds y = TTb(ntuple(i -> Dual{Tg}(fdata[i], diffdata[i]), length(fdata)))
    return y
end

function _insert_full_gradient(f::Vec{dim}, dfdx::Tensor{2,dim}, ::Tg) where{dim, Tg}
    fdata = get_data(f)
    diffdata = get_data(dfdx)
    @inbounds y = Vec{dim}(i -> Dual{Tg}(fdata[i], ntuple(j->diffdata[i+dim*(j-1)], dim)))
    return y
end

function _insert_full_gradient(f::Tensor{2,dim,<:Any,N}, dfdx::Tensor{4,dim}, ::Tg) where{dim, N, Tg}
    fdata = get_data(f)
    diffdata = get_data(dfdx)
    @inbounds y = Tensor{2,dim}(ntuple(i->Dual{Tg}(fdata[i], ntuple(j->diffdata[i+N*(j-1)],N)), N))
    return y
end
function _insert_full_gradient(f::SymmetricTensor{2,dim,<:Any,N}, dfdx::SymmetricTensor{4,dim}, ::Tg) where{dim, N, Tg}
    fdata = get_data(f)
    diffdata = get_data(dfdx)
    @inbounds y = SymmetricTensor{2,dim}(ntuple(i->Dual{Tg}(fdata[i], ntuple(j->diffdata[i+N*(j-1)],N)), N))
    return y
end


##################
# Load functions #
##################

# Loaders are supposed to take a tensor of real values and convert it
# into a tensor of dual values where the seeds are correctly defined.

@inline function _load(v::Number, ::Tg) where Tg
    return Dual{Tg}(v, one(v))
end

@inline function _load(v::Vec{1, T}, ::Tg) where {T, Tg}
    @inbounds v_dual = Vec{1}((Dual{Tg}(v[1], one(T)),))
    return v_dual
end

@inline function _load(v::Vec{2, T}, ::Tg) where {T, Tg}
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{2}((Dual{Tg}(v[1], o, z),
                               Dual{Tg}(v[2], z, o)))
    return v_dual
end

@inline function _load(v::Vec{3, T}, ::Tg) where {T, Tg}
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{3}((Dual{Tg}(v[1], o, z, z),
                               Dual{Tg}(v[2], z, o, z),
                               Dual{Tg}(v[3], z, z, o)))
    return v_dual
end

# Second order tensors
@inline function _load(v::Tensor{2, 1, T}, ::Tg) where {T, Tg}
    @inbounds v_dual = Tensor{2, 1}((Dual{Tg}(get_data(v)[1], one(T)),))
    return v_dual
end

@inline function _load(v::SymmetricTensor{2, 1, T}, ::Tg) where {T, Tg}
    @inbounds v_dual = SymmetricTensor{2, 1}((Dual{Tg}(get_data(v)[1], one(T)),))
    return v_dual
end

@inline function _load(v::Tensor{2, 2, T}, ::Tg) where {T, Tg}
    data = get_data(v)
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Tensor{2, 2}((Dual{Tg}(data[1], o, z, z, z),
                                     Dual{Tg}(data[2], z, o, z, z),
                                     Dual{Tg}(data[3], z, z, o, z),
                                     Dual{Tg}(data[4], z, z, z, o)))
    return v_dual
end

@inline function _load(v::SymmetricTensor{2, 2, T}, ::Tg) where {T, Tg}
    data = get_data(v)
    o = one(T)
    o2 = convert(T, 1/2)
    z = zero(T)
    @inbounds v_dual = SymmetricTensor{2, 2}((Dual{Tg}(data[1], o, z, z),
                                              Dual{Tg}(data[2], z, o2, z),
                                              Dual{Tg}(data[3], z, z, o)))
    return v_dual
end

@inline function _load(v::Tensor{2, 3, T}, ::Tg) where {T, Tg}
    data = get_data(v)
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Tensor{2, 3}((Dual{Tg}(data[1], o, z, z, z, z, z, z, z, z),
                                     Dual{Tg}(data[2], z, o, z, z, z, z, z, z, z),
                                     Dual{Tg}(data[3], z, z, o, z, z, z, z, z, z),
                                     Dual{Tg}(data[4], z, z, z, o, z, z, z, z, z),
                                     Dual{Tg}(data[5], z, z, z, z, o, z, z, z, z),
                                     Dual{Tg}(data[6], z, z, z, z, z, o, z, z, z),
                                     Dual{Tg}(data[7], z, z, z, z, z, z, o, z, z),
                                     Dual{Tg}(data[8], z, z, z, z, z, z, z, o, z),
                                     Dual{Tg}(data[9], z, z, z, z, z, z, z, z, o)))
    return v_dual
end

@inline function _load(v::SymmetricTensor{2, 3, T}, ::Tg) where {T, Tg}
    data = get_data(v)
    o = one(T)
    o2 = convert(T, 1/2)
    z = zero(T)
    @inbounds v_dual = SymmetricTensor{2, 3}((Dual{Tg}(data[1], o, z, z, z, z, z),
                                              Dual{Tg}(data[2], z, o2, z, z, z, z),
                                              Dual{Tg}(data[3], z, z, o2, z, z, z),
                                              Dual{Tg}(data[4], z, z, z, o, z, z),
                                              Dual{Tg}(data[5], z, z, z, z, o2, z),
                                              Dual{Tg}(data[6], z, z, z, z, z, o)))
    return v_dual
end

"""
    gradient(f::Function, v::Union{SecondOrderTensor, Vec, Number})
    gradient(f::Function, v::Union{SecondOrderTensor, Vec, Number}, :all)

Computes the gradient of the input function. If the (pseudo)-keyword `all`
is given, the value of the function is also returned as a second output argument.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇f = gradient(norm, A)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.374672  0.63107
 0.63107   0.25124

julia> ∇f, f = gradient(norm, A, :all);
```
"""
function gradient(f::F, v::V) where {F, V <: Union{SecondOrderTensor, Vec, Number}}
    v_dual = _load(v, Tag(f, V))
    res = f(v_dual)
    return _extract_gradient(res, v)
end
function gradient(f::F, v::V, ::Symbol) where {F, V <: Union{SecondOrderTensor, Vec, Number}}
    v_dual = _load(v, Tag(f, V))
    res = f(v_dual)
    return _extract_gradient(res, v), _extract_value(res)
end
const ∇ = gradient

"""
    hessian(f::Function, v::Union{SecondOrderTensor, Vec, Number})
    hessian(f::Function, v::Union{SecondOrderTensor, Vec, Number}, :all)

Computes the hessian of the input function. If the (pseudo)-keyword `all`
is given, the lower order results (gradient and value) of the function is
also returned as a second and third output argument.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇∇f = hessian(norm, A)
2×2×2×2 SymmetricTensor{4, 2, Float64, 9}:
[:, :, 1, 1] =
  0.988034  -0.271765
 -0.271765  -0.108194

[:, :, 2, 1] =
 -0.271765   0.11695
  0.11695   -0.182235

[:, :, 1, 2] =
 -0.271765   0.11695
  0.11695   -0.182235

[:, :, 2, 2] =
 -0.108194  -0.182235
 -0.182235   1.07683

julia> ∇∇f, ∇f, f = hessian(norm, A, :all);
```
"""
function hessian(f::F, v::Union{SecondOrderTensor, Vec, Number}) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v)
end

function hessian(f::F, v::Union{SecondOrderTensor, Vec, Number}, ::Symbol) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v), gradient(f, v, :all)...
end
const ∇∇ = hessian

"""
    divergence(f, x)

Calculate the divergence of the vector field `f`, in the point `x`.

# Examples
```jldoctest
julia> f(x) = 2x;

julia> x = rand(Vec{3});

julia> divergence(f, x)
6.0
```
"""
divergence(f::F, v::Vec) where {F<:Function} = tr(gradient(f, v))

"""
    curl(f, x)

Calculate the curl of the vector field `f`, in the point `x`.

# Examples
```jldoctest
julia> f(x) = Vec{3}((x[2], x[3], -x[1]));

julia> x = rand(Vec{3});

julia> curl(f, x)
3-element Vec{3, Float64}:
 -1.0
  1.0
 -1.0
```
"""
function curl(f::F, v::Vec{3}) where F
    @inbounds begin
        ∇f = gradient(f, v)
        c = Vec{3}((∇f[3,2] - ∇f[2,3], ∇f[1,3] - ∇f[3,1], ∇f[2,1] - ∇f[1,2]))
    end
    return c
end
curl(f::F, v::Vec{1, T}) where {F, T} = curl(f, Vec{3}((v[1], T(0), T(0))))
curl(f::F, v::Vec{2, T}) where {F, T} = curl(f, Vec{3}((v[1], v[2], T(0))))

"""
    laplace(f, x)

Calculate the laplacian of the field `f`, in the point `x`.
If `f` is a vector field, use broadcasting.

# Examples
```jldoctest
julia> x = rand(Vec{3});

julia> f(x) = norm(x);

julia> laplace(f, x)
2.9633756571179273

julia> g(x) = x*norm(x);

julia> laplace.(g, x)
3-element Vec{3, Float64}:
 1.9319830062026155
 3.2540895437409754
 1.2955087437219237
```
"""
function laplace(f::F, v) where F
    return divergence(x -> gradient(f, x), v)
end
const Δ = laplace

function Broadcast.broadcasted(::typeof(laplace), f::F, v::V) where {F, V <: Vec{3}}
    @inbounds begin
        tag = Tag(f, V)
        vdd = _load(_load(v, tag), tag)
        res = f(vdd)
        v1 = res[1].partials[1].partials[1] + res[1].partials[2].partials[2] + res[1].partials[3].partials[3]
        v2 = res[2].partials[1].partials[1] + res[2].partials[2].partials[2] + res[2].partials[3].partials[3]
        v3 = res[3].partials[1].partials[1] + res[3].partials[2].partials[2] + res[3].partials[3].partials[3]
    end
    return Vec{3}((v1, v2, v3))
end
