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

# Applies the function f to all linear indices: TT((f(1), f(2), ..., f(M))).
# (Base's tuple `map` is not usable here: it leaves the unrolled path beyond
# 32 components, e.g. Tensor{4, 3} with 81.)
@inline function apply_all(::Type{TT}, f::F) where {TT <: Union{Tensor, SymmetricTensor, MixedTensor}, F <: Function}
    B = get_base(TT)
    return B(ntuple(f, Val(n_components(B))))
end
@inline apply_all(S::AbstractTensor, f::F) where {F <: Function} = apply_all(get_base(typeof(S)), f)

# map implementations
@inline function _map(f, S::AbstractTensor)
    return apply_all(S, @inline function(i) @inbounds f(S.data[i]); end)
end

# the caller of 2 arg _map MUST guarantee that both arguments have
# the same base (Tensor{order, dim} / SymmetricTensor{order, dim}) but not necessarily the same eltype
@inline function _map(f, S1::AbstractTensor, S2::AbstractTensor)
    return apply_all(S1, @inline function(i) @inbounds f(S1.data[i], S2.data[i]); end)
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
