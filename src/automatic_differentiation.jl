import ForwardDiff: Dual, partials, value, Tag

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
# AbstractTensor output -> AbstractTensor value
@inline _extract_value(v::AbstractTensor{<:Any,<:Any,<:Dual}) = _map(value, v)

#######################
# Gradient extraction #
#######################

# Scalar output, Scalar input -> Scalar gradient
@inline function _extract_gradient(v::Dual, ::Number)
    return @inbounds partials(v)[1]
end
# Vec, Tensor{2/4}, SymmetricTensor{2/4} output, Scalar input -> Vec, Tensor{2/4}, SymmetricTensor{2/4} gradient
@inline function _extract_gradient(v::AbstractTensor{<:Any,<:Any,<:Dual}, ::Number)
    return _map(@inline(d -> @inbounds(partials(d)[1])), v)
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
        p = map(partials, v.data)
        @inbounds return $TT($expr)
    end
end

# SymmetricTensor{2} output, SymmetricTensor{2} input -> SymmetricTensor{4} gradient
@generated function _extract_gradient_dual(v::SymmetricTensor{2, dim, <:Dual}, ::SymmetricTensor{2, dim}) where {dim}
    N = n_components(SymmetricTensor{2, dim})
    expr = Expr(:tuple)
    for j in 1:N, i in 1:N
        push!(expr.args, :(p[$i][$j]))
    end
    return quote
        $(Expr(:meta, :inline))
        p = map(partials, v.data)
        @inbounds return $(SymmetricTensor{4, dim})($expr)
    end
end

# for non dual variable
@inline function _extract_value(v::Any)
    return v
end
@inline function _extract_gradient(::T, x::TT) where {T <: Real, TT <: AbstractTensor}
    zero(get_base(TT){T})
end
# symmetric input and output: keep the zero gradient symmetric
@generated function _extract_gradient_nondual(v::SymmetricTensor{order, dim, T}, ::SymmetricTensor{order, dim}) where {T<:Real, order, dim}
    RetType = SymmetricTensor{order+order, dim, T}
    return quote
        $(Expr(:meta, :inline))
        zero($RetType)
    end
end
# all other shapes (e.g. a constant Vec{2}-valued function of a Vec{3}): the
# zero of the same gradient type the dual-valued path produces — a MixedTensor
# over the concatenated dimensions, collapsed to a regular Tensor when they
# all agree
@generated function _extract_gradient_nondual(v::AbstractTensor{<:Any, <:Any, T}, x::AbstractTensor) where {T <: Real}
    dims = (base_size(get_base(v))..., base_size(get_base(x))...)
    RetType = regular_if_possible(MixedTensor{length(dims), Tuple{dims...}, T, prod(dims)})
    return quote
        $(Expr(:meta, :inline))
        zero($RetType)
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
    return :($(esc(f))(x :: Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual}) = propagate_gradient($(esc(f_dfdx)), x))
end

"""
    propagate_gradient(f_dfdx::Function, x, args...)
    propagate_gradient(f_dfdx::Function, ::Val{i}, args...)
    propagate_gradient(f_dfdx::Function, ::Val{(i, j, ...)}, args...)

The building block of [`@implement_gradient`](@ref): propagate an analytical
derivative through automatic differentiation. `f_dfdx` must return the tuple
`(f(args...), ∂f∂x(args...))`, where `x` is the differentiation argument —
the first argument, or `args[i]` in the `Val{i}` form. It is called with `x`
replaced by its primal value (one level of `ForwardDiff.Dual` stripped); all
other arguments are passed through unchanged.

When several arguments depend on the differentiated variable, list them all:
with `Val((i, j, ...))`, `f_dfdx` is called with each of `args[i], args[j], ...`
peeled and must return `(f(args...), (∂f∂args[i], ∂f∂args[j], ...))`; the
contributions are summed according to the chain rule. Leaving such an
argument out of the active set silently drops its contribution — e.g.
`g(F, C)` evaluated at `C = tdot(F)` inside a `gradient` call needs
`Val((1, 2))`, not `Val(1)`.

The analytical Jacobian is applied directly to the partial derivatives
carried by `x`, so this works under any outer differentiation context
(`Tensors.gradient`/`hessian`, plain `ForwardDiff`, or nested duals), and the
symmetric-tensor gradient convention is inherited from the incoming seeds.
If `x` carries no dual numbers, the primal value is returned unchanged.

Use this instead of `@implement_gradient` when the function takes additional
non-differentiated arguments, or when the analytical-derivative method should
be more specific than the broad signature the macro defines:

```julia
f(x::SymmetricTensor{2, dim, <:Dual}, p) where {dim} = propagate_gradient(f_dfdx, x, p)
g(p, x::SymmetricTensor{2, dim, <:Dual}) where {dim} = propagate_gradient(g_dgdx, Val(2), p, x)
```
"""
@inline function propagate_gradient(f_dfdx::F, x::Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual}, args...) where {F <: Function}
    return propagate_gradient(f_dfdx, Val(1), x, args...)
end

# One active argument, `Val(i)`, or several, `Val((i, j, ...))`. In the
# multi-argument form `f_dfdx` is called with every active argument peeled
# and must return `(y, (∂y∂args[i], ∂y∂args[j], ...))` with the Jacobians in
# the same order as the indices. The chain rule sums the contribution of each
# active argument's incoming derivative lanes:
#
#     dy/dx = ∂f∂args[i] ⊙ dargs[i]/dx + ∂f∂args[j] ⊙ dargs[j]/dx + ...
#
# All dual-carrying active arguments must share the same (outermost)
# differentiation tag — mixing lanes of unrelated differentiations would be
# meaningless, so that is an error. Active arguments that carry no dual
# numbers contribute nothing (their term of the sum is dropped).
@inline function propagate_gradient(f_dfdx::F, ::Val{i}, args...) where {F <: Function, i}
    # `i` is a compile-time constant: this branch folds away
    if i isa Integer
        x = args[i]
        fval, dfdx_val = f_dfdx(Base.setindex(args, extract_value(x), i)...)
        return _insert_dual(fval, dfdx_val, x)
    elseif i isa Tuple{Vararg{Integer}}
        xs = ntuple(k -> args[i[k]], Val(length(i)))
        _check_common_tag(xs)
        fval, dfdx_vals = f_dfdx(_peel(Val(i), args...)...)
        length(dfdx_vals) == length(i) ||
            throw(ArgumentError("f_dfdx must return one Jacobian per active argument, got $(length(dfdx_vals)) for $(length(i))"))
        fdual = _chain_sum(dfdx_vals, xs)
        return fdual === nothing ? fval : _replace_value(fval, fdual)
    else
        throw(ArgumentError("the Val argument of propagate_gradient must hold an integer or a tuple of integers"))
    end
end

# `args` with every active argument replaced by its primal value
@generated function _peel(::Val{actives}, args...) where {actives}
    peeled = [k in actives ? :(extract_value(args[$k])) : :(args[$k]) for k in 1:length(args)]
    return :(tuple($(peeled...)))
end

# the outermost differentiation tag of a dual-carrying value, or `nothing`
_tag_of(::Dual{Tg}) where {Tg} = Tg
_tag_of(::AbstractTensor{<:Any, <:Any, <:Dual{Tg}}) where {Tg} = Tg
_tag_of(::Any) = nothing

@inline function _check_common_tag(xs::Tuple)
    tag = nothing
    for x in xs
        tx = _tag_of(x)
        if tx !== nothing
            if tag !== nothing && tag !== tx
                throw(ArgumentError("the active arguments carry dual numbers with different differentiation tags; they must come from the same differentiation call"))
            end
            tag = tx
        end
    end
    return nothing
end

# Σₖ ∂y∂xₖ ⊙ xₖ over the dual-carrying active arguments (in argument order),
# in Dual arithmetic; `nothing` when no active argument carries duals
@inline _chain_sum(dfdxs::Tuple{}, xs::Tuple{}) = nothing
@inline function _chain_sum(dfdxs::Tuple, xs::Tuple)
    rest = _chain_sum(Base.tail(dfdxs), Base.tail(xs))
    _tag_of(xs[1]) === nothing && return rest
    c = _chain(dfdxs[1], xs[1])
    return rest === nothing ? c : c + rest
end

"""
    extract_value(x)

Return the primal value of `x`: strips one level of `ForwardDiff.Dual` numbers
from a `Dual` or from a tensor with `Dual` entries. Other values are returned
unchanged. Useful e.g. for storing state variables inside a function that is
being differentiated.
"""
extract_value(x) = _extract_value(x)

# backwards-compatible internal name (the macro used to call this directly)
_propagate_gradient(f_dfdx::Function, x::Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual}) = propagate_gradient(f_dfdx, x)

# Define the _insert_gradient method
"""
    _insert_gradient(f::Union{Number,AbstractTensor}, dfdg::Union{Number,AbstractTensor}, g::Union{ForwardDiff.Dual, AbstractTensor{<:Any,<:Any,<:ForwardDiff.Dual}})

Insert an analytical gradient for use with automatic differentiation: given
the primal value `f`, the analytical derivative `dfdg`, and the dual-carrying
input `g`, return `f` as a dual number/tensor with the correctly propagated
partial derivatives. Prefer the public [`propagate_gradient`](@ref), which
wraps this:
```julia
f(g::Tensor{2,dim,<:ForwardDiff.Dual}) where {dim} = propagate_gradient(f_dfdg, g)
```

Implementation: contracting the Jacobian with the dual-valued input applies
the chain rule to every incoming partial lane at once (the result's partials
are exactly `∂f∂g` applied to each lane); the value part is then replaced
with the exact primal. The original differentiation input is never
reconstructed from the `Tag` type parameters, so any outer context (Tensors'
own operators, plain ForwardDiff, nested duals) works, and the symmetric
½-seed convention is inherited from the incoming lanes. Arguments other than
the differentiation argument must be passive with respect to the peeled tag.
"""
_insert_gradient(f::Union{Number, AbstractTensor}, dfdg::Union{Number, AbstractTensor},
                 g::Union{AbstractTensor{<:Any, <:Any, <:Dual}, Dual}) = _insert_dual(f, dfdg, g)

@inline function _insert_dual(fval::Union{Number, AbstractTensor}, dfdx::Union{Number, AbstractTensor},
                              x::Union{Dual{Tg}, AbstractTensor{<:Any, <:Any, <:Dual{Tg}}}) where {Tg}
    # one contraction in Dual arithmetic: the partial lanes of the result are
    # exactly ∂f∂x applied to each incoming lane; its value part is then
    # replaced with the (exact) primal value
    fdual = _chain(dfdx, x)
    return _replace_value(fval, fdual)
end
# the differentiation argument carries no duals: nothing to insert
@inline _insert_dual(fval::Union{Number, AbstractTensor}, dfdx, x) = fval

# chain rule: contract the Jacobian over all indices of the input
@inline _chain(dfdx::Union{Number, AbstractTensor}, x::Number) = dfdx * x
@inline _chain(dfdx::AbstractTensor, x::Vec) = dfdx ⋅ x
@inline _chain(dfdx::AbstractTensor, x::SecondOrderTensor) = dfdx ⊡ x

@inline _replace_value(fval::Number, fdual::Dual{Tg}) where {Tg} = Dual{Tg}(fval, ForwardDiff.partials(fdual))
@inline function _replace_value(fval::AbstractTensor, fdual::AbstractTensor{<:Any, <:Any, <:Dual{Tg}}) where {Tg}
    if get_base(typeof(fval)) !== get_base(typeof(fdual))
        throw(ArgumentError(string("the type of the analytical derivative is inconsistent with the function value: ",
                                   "the chain rule produced a ", get_base(typeof(fdual)), " for a ", get_base(typeof(fval)),
                                   " function value (all tensors must be symmetric, or none)")))
    end
    return _map(@inline((v, d) -> Dual{Tg}(v, ForwardDiff.partials(d))), fval, fdual)
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
        data = v.data
        @inbounds return $TB($expr)
    end
end

# Second order symmetric tensors: diagonal components are seeded with 1,
# off-diagonal (paired) components with 1/2
@generated function _load(v::SymmetricTensor{2, dim, T}, ::Tg) where {dim, T, Tg}
    N = n_components(SymmetricTensor{2, dim})
    # compact storage is the lower triangle, column-major
    isdiag = [i == j for j in 1:dim for i in j:dim]
    expr = Expr(:tuple)
    for a in 1:N
        seeds = Expr(:tuple)
        for b in 1:N
            push!(seeds.args, a == b ? (isdiag[a] ? :o : :o2) : :z)
        end
        push!(expr.args, :(Dual{Tg}(data[$a], $seeds)))
    end
    return quote
        $(Expr(:meta, :inline))
        data = v.data
        o = one(T)
        o2 = convert(T, 1/2)
        z = zero(T)
        @inbounds return $(SymmetricTensor{2, dim})($expr)
    end
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
For a three-dimensional vector field, use broadcasting (`laplace.(f, x)`
with `x::Vec{3}`); vector fields of other dimensions are not supported.

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
