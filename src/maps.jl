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

# Sum-of-products reduction over two equal-length expression lists, collapsing
# duplicate products into integer factors (generation-time helper; this is what
# turns a full-grid read of packed symmetric data into `2 * x * y` terms).
function reducer(ex1i, ex2i, madd = false)
    ex1, ex2 = remove_duplicates(ex1i, ex2i)
    N = length(ex1)
    expr = :($(ex1[1]) * $(ex2[1]))
    for i in 2:N
        expr = madd ? :(muladd($(ex1[i]), $(ex2[i]), $expr)) :
                      :($(expr) + $(ex1[i]) * $(ex2[i]))
    end
    return expr
end

function remove_duplicates(ex1in, ex2in)
    ex1out, ex2out = Expr[], Expr[]
    exout = Expr[]
    factors = ones(Int, length(ex1in))
    for (ex1ine, ex2ine) in zip(ex1in, ex2in)
        prod = :($ex1ine * $ex2ine)
        i = findfirst(isequal(prod), exout) # check if this product exists in the output
        if i === nothing # this product does not exist yet
            push!(ex1out, ex1ine)
            push!(ex2out, ex2ine)
            push!(exout, prod)
        else # found a duplicate
            factors[i] += 1
        end
    end
    for i in 1:length(ex1out)
        factors[i] != 1 && (ex1out[i] = :($(factors[i]) * $(ex1out[i])))
    end
    return ex1out, ex2out
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
