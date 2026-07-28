import ForwardDiff: Dual, partials, value, Tag
# The HyperHessiansForwardDiffExt extension (loaded since Tensors depends on
# both packages) resolves HyperDual/Dual ambiguities for nested AD.
import HyperHessians: HyperDual

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

@inline function _extract_gradient(v::Dual, ::TT) where {TT <: AbstractTensor}
    return get_base(TT)(partials(v).values)
end

@inline _extract_gradient(v::AbstractTensor{<:Any, <:Any, <:Dual}, u::AbstractTensor) = _extract_gradient_dual(v, u)
@inline _extract_gradient(v::AbstractTensor, u::AbstractTensor) = _extract_gradient_nondual(v, u)

@inline _extract_gradient_dual(v::AbstractTensor{<:Any, <:Any, <:Dual}, u::AbstractTensor) = _extract_gradient(makemixed(v), makemixed(u))

@generated function _extract_gradient_dual(v::MixedTensor{order1, dims1, <:Dual}, u::MixedTensor{order2, dims2}) where {order1, dims1, order2, dims2}
    expr = Expr(:tuple)
    N1 = n_components(MixedTensor{order1, dims1})
    N2 = n_components(MixedTensor{order2, dims2})
    for j = 1:N2
        for i = 1:N1
            push!(expr.args, :(p[$i][$j]))
        end
    end
    TT = regular_if_possible(MixedTensor{order1 + order2, Tuple{size(v)..., size(u)...}})
    return quote
        $(Expr(:meta, :inline))
        p = map(partials, get_data(v))
        @inbounds return $TT($expr)
    end
end

# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient_dual(v::SymmetricTensor{2, 1, <: Dual}, ::SymmetricTensor{2, 1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = SymmetricTensor{4, 1}((p1[1],))
    end
    return ∇f
end

# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient_dual(v::SymmetricTensor{2, 2, <: Dual}, ::SymmetricTensor{2, 2})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[2,2])
        ∇f = SymmetricTensor{4, 2}((p1[1], p2[1], p3[1],
                                    p1[2], p2[2], p3[2],
                                    p1[3], p2[3], p3[3]))
    end
    return ∇f
end

# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@inline function _extract_gradient_dual(v::SymmetricTensor{2, 3, <: Dual}, ::SymmetricTensor{2, 3})
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

# for non dual variable
@inline function _extract_value(v::Any)
    return v
end
@inline function _extract_gradient(::T, x::TT) where {T <: Real, TT <: AbstractTensor}
    zero(get_base(TT){T})
end
for TensorType in (Tensor, SymmetricTensor)
    @eval begin
        @generated function _extract_gradient_nondual(v::$TensorType{order, dim, T}, ::$TensorType{order, dim}) where {T<:Real, order, dim}
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
# Scalar
@inline function _load(v::Number, ::Tg) where Tg
    return Dual{Tg}(v, one(v))
end

# Any order non-symmetric tensor
@generated function _load(v::Union{MixedTensor{<:Any, <:Any, T}, Tensor{<:Any, <:Any, T}}, ::Tg) where {T, Tg}
    TB = get_base(v)
    N = n_components(TB)
    function makedual(i)
        partials = Expr(:tuple)
        foreach(j -> push!(partials.args, j == i ? :(one(T)) : :(zero(T))), 1:N)
        return :(Dual{Tg}(data[$i], $partials))
    end
    expr = Expr(:tuple)
    for i = 1:N
        push!(expr.args, makedual(i))
    end
    return quote
        $(Expr(:meta, :inline))
        data = get_data(v)
        @inbounds return $TB($expr)
    end
end

# Second order symmetric tensors
@inline function _load(v::SymmetricTensor{2, 1, T}, ::Tg) where {T, Tg}
    @inbounds v_dual = SymmetricTensor{2, 1}((Dual{Tg}(get_data(v)[1], one(T)),))
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

###################
# Hessian loaders #
###################

# Loaders for single-pass Hessian computation with HyperHessians.HyperDual.
# Both ϵ1 and ϵ2 are seeded with the same (unit, or half for symmetric
# off-diagonals) seeds so that one function evaluation carries the value (v),
# gradient (ϵ1) and full Hessian (ϵ12).

@inline function _load_hessian(v::Number)
    o = one(v)
    return HyperDual(v, (o,), (o,))
end

@generated function _load_hessian(v::Union{MixedTensor{2, <:Any, T}, Tensor{2, <:Any, T}, Vec{<:Any, T}}) where {T}
    TB = get_base(v)
    N = n_components(TB)
    function makehyper(i)
        seed = Expr(:tuple, [j == i ? :o : :z for j in 1:N]...)
        return :(HyperDual(data[$i], $seed, $seed))
    end
    expr = Expr(:tuple, [makehyper(i) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        data = get_data(v)
        o = one(T); z = zero(T)
        @inbounds return $TB($expr)
    end
end

@generated function _load_hessian(v::SymmetricTensor{2, dim, T}) where {dim, T}
    N = n_components(SymmetricTensor{2, dim})
    isdiag = Bool[i == j for j in 1:dim for i in j:dim]
    function makehyper(i)
        seed = Expr(:tuple, [j == i ? (isdiag[i] ? :o : :h) : :z for j in 1:N]...)
        return :(HyperDual(data[$i], $seed, $seed))
    end
    expr = Expr(:tuple, [makehyper(i) for i in 1:N]...)
    return quote
        $(Expr(:meta, :inline))
        data = get_data(v)
        o = one(T); h = convert(T, 1/2); z = zero(T)
        @inbounds return $(SymmetricTensor{2, dim})($expr)
    end
end

######################
# Hessian extraction #
######################

# Scalar output -> the Hessian is in the ϵ12 components.
@inline _extract_hessian(f::F, r::HyperDual, ::Number) where {F} = @inbounds r.ϵ12[1][1]

# Since ϵ1 and ϵ2 carry identical seeds the ϵ12 matrix is symmetric, so the
# extraction reads ϵ12[j][i], which walks the nested tuples in memory order.
@generated function _extract_hessian(f::F, r::HyperDual, v::Union{Vec, Tensor{2}, MixedTensor{2}}) where {F}
    N = n_components(get_base(v))
    TT = regular_if_possible(MixedTensor{2 * length(size(v)), Tuple{size(v)..., size(v)...}})
    expr = Expr(:tuple)
    for j in 1:N, i in 1:N
        push!(expr.args, :(ϵ12[$j][$i]))
    end
    return quote
        $(Expr(:meta, :inline))
        ϵ12 = r.ϵ12
        @inbounds return $TT($expr)
    end
end

@generated function _extract_hessian(f::F, r::HyperDual, v::SymmetricTensor{2, dim}) where {F, dim}
    N = n_components(SymmetricTensor{2, dim})
    expr = Expr(:tuple)
    for j in 1:N, i in 1:N
        push!(expr.args, :(ϵ12[$j][$i]))
    end
    return quote
        $(Expr(:meta, :inline))
        ϵ12 = r.ϵ12
        @inbounds return $(SymmetricTensor{4, dim})($expr)
    end
end

# AbstractTensor output, scalar input -> Hessian of each component.
@generated function _extract_hessian(f::F, r::AbstractTensor{<:Any, <:Any, <:HyperDual}, ::Number) where {F}
    TB = get_base(r)
    expr = Expr(:tuple, [:(get_data(r)[$i].ϵ12[1][1]) for i in 1:n_components(TB)]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TB($expr)
    end
end

# Output without HyperDual sensitivity: f is constant, zero Hessian.
# Signatures for v mirror the HyperDual methods exactly so that dispatch is
# resolved by the specificity of r (HyperDual <: Real).
@inline _extract_hessian(f::F, r::Real, ::Number) where {F} = zero(r)
@generated function _extract_hessian(f::F, r::Real, v::Union{Vec, Tensor{2}, MixedTensor{2}}) where {F}
    TT = regular_if_possible(MixedTensor{2 * length(size(v)), Tuple{size(v)..., size(v)...}})
    return quote
        $(Expr(:meta, :inline))
        zero($TT{typeof(r)})
    end
end
@inline _extract_hessian(f::F, r::Real, ::SymmetricTensor{2, dim}) where {F, dim} = zero(SymmetricTensor{4, dim, typeof(r)})

# Constant AbstractTensor output with scalar input: zero Hessian.
@inline _extract_hessian(f::F, r::AbstractTensor, ::Number) where {F} = zero(r)

# AbstractTensor output with tensor input is not supported by the single-pass
# HyperDual path; fall back to nested dual differentiation.
@inline function _extract_hessian(f::F, r::AbstractTensor, v::Union{SecondOrderTensor, Vec}) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v)
end

# Gradient (ϵ1) extraction from a HyperDual result.
@inline _extract_hessian_gradient(r::HyperDual, ::Number) = @inbounds r.ϵ1[1]

@generated function _extract_hessian_gradient(r::HyperDual, v::Union{Vec, SecondOrderTensor})
    TB = get_base(v)
    expr = Expr(:tuple, [:(ϵ1[$i]) for i in 1:n_components(TB)]...)
    return quote
        $(Expr(:meta, :inline))
        ϵ1 = r.ϵ1
        @inbounds return $TB($expr)
    end
end

@generated function _extract_hessian_gradient(r::AbstractTensor{<:Any, <:Any, <:HyperDual}, ::Number)
    TB = get_base(r)
    expr = Expr(:tuple, [:(get_data(r)[$i].ϵ1[1]) for i in 1:n_components(TB)]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TB($expr)
    end
end

# Value extraction from a HyperDual result.
@inline _extract_value(r::HyperDual) = r.v

@generated function _extract_value(r::AbstractTensor{<:Any, <:Any, <:HyperDual})
    TB = get_base(r)
    expr = Expr(:tuple, [:(get_data(r)[$i].v) for i in 1:n_components(TB)]...)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TB($expr)
    end
end

# hessian(f, v, :all) extraction: (hessian, gradient, value)
@inline function _extract_hessian_all(f::F, r::HyperDual, v::Union{SecondOrderTensor, Vec}) where {F}
    return _extract_hessian(f, r, v), _extract_hessian_gradient(r, v), _extract_value(r)
end
@inline function _extract_hessian_all(f::F, r::HyperDual, v::Number) where {F}
    return _extract_hessian(f, r, v), _extract_hessian_gradient(r, v), _extract_value(r)
end
@inline function _extract_hessian_all(f::F, r::AbstractTensor{<:Any, <:Any, <:HyperDual}, v::Number) where {F}
    return _extract_hessian(f, r, v), _extract_hessian_gradient(r, v), _extract_value(r)
end
@inline function _extract_hessian_all(f::F, r::Real, v::Union{SecondOrderTensor, Vec}) where {F}
    return _extract_hessian(f, r, v), _extract_gradient(r, v), r
end
@inline function _extract_hessian_all(f::F, r::Real, v::Number) where {F}
    return zero(r), zero(r), r
end
@inline function _extract_hessian_all(f::F, r::AbstractTensor, v::Number) where {F}
    return zero(r), zero(r), r
end
@inline function _extract_hessian_all(f::F, r::AbstractTensor, v::Union{SecondOrderTensor, Vec}) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v), gradient(f, v, :all)...
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
function gradient(f::F, v::V) where {F, V <: Union{SecondOrderTensor, Vec, Number, MixedTensor}}
    v_dual = _load(v, Tag(f, V))
    res = f(v_dual)
    return _extract_gradient(res, v)
end
function gradient(f::F, v::V, ::Symbol) where {F, V <: Union{SecondOrderTensor, Vec, Number, MixedTensor}}
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

For scalar-valued functions (and tensor-valued functions of a scalar argument)
the hessian (and gradient and value) is computed in a single function evaluation
using hyper-dual numbers from HyperHessians.jl. Note that analytical derivatives
supplied with [`@implement_gradient`](@ref) dispatch on `ForwardDiff.Dual` and
are therefore not used by `hessian`; such functions are differentiated by
hyper-dual arithmetic (and error if their primal methods do not accept
generic `Real` arguments).

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
    res = f(_load_hessian(v))
    return _extract_hessian(f, res, v)
end

function hessian(f::F, v::Union{SecondOrderTensor, Vec, Number}, ::Symbol) where {F}
    res = f(_load_hessian(v))
    return _extract_hessian_all(f, res, v)
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
function curl(f::F, v::Vec{2}) where {F}
    @inbounds begin
        ∇f = gradient(f, v)
        c = Vec{3}((zero(eltype(∇f)), zero(eltype(∇f)), ∇f[2,1] - ∇f[1,2]))
    end
    return c
end
function curl(f::F, v::Vec{1, T}) where {F, T}
    return zero(Vec{3, eltype(f(v))}) / oneunit(T)
end

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
