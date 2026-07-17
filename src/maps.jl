##################################################################
# Component-map layer                                            #
#                                                                #
# Expression builders over the independent components of a       #
# tensor type, and the `apply_all`/`_map` helpers built on them. #
# Everything element-wise (conversion, ±, scalar ops, zero/one/  #
# rand/fill, from-function construction, ...) goes through here, #
# for all three tensor kinds.                                    #
##################################################################

# Tuple expression `(f(inds_1...), f(inds_2...), ...)` over the independent
# components (in storage order) of the base type `TT`.
function component_expr(TT::Type, f)
    return Expr(:tuple, [f(c...) for c in base_components(TT)]...)
end

# Tuple expression `(f(1), f(2), ..., f(M))` over the stored components of `TT`.
function linear_expr(TT::Type, f)
    return Expr(:tuple, [f(i) for i in 1:n_components(TT)]...)
end

# Applies the function f to all indices f(1), f(2), ... f(n_independent_components)
@generated function apply_all(S::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}, Type{MixedTensor{order, dim}}}, f::Function) where {order, dim}
    TensorType = get_base(get_type(S))
    exp = linear_expr(TensorType, i -> :(f($i)))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $TensorType($exp)
    end
end

@inline function apply_all(S::Union{Tensor{order, dim}, SymmetricTensor{order, dim}, MixedTensor{order, dim}}, f::Function) where {order, dim}
    apply_all(get_base(typeof(S)), f)
end

# map implementations
@inline function _map(f, S::AbstractTensor)
    return apply_all(S, @inline function(i) @inbounds f(S.data[i]); end)
end

# the caller of 2 arg _map MUST guarantee that both arguments have
# the same base (Tensor{order, dim} / SymmetricTensor{order, dim}) but not necessarily the same eltype
@inline function _map(f, S1::AbstractTensor, S2::AbstractTensor)
    return apply_all(S1, @inline function(i) @inbounds f(get_data(S1)[i], get_data(S2)[i]); end)
end

# strip Unitful-like units if necessary (used by eigen)
function ustrip(S::SymmetricTensor{order, dim, T}) where {order, dim, T}
    ou = oneunit(T)
    if typeof(ou / ou) === T # no units
        return S
    else # units, so strip them by dividing with oneunit(T)
        return SymmetricTensor{order, dim}(map(x -> x / ou, S.data))
    end
end
