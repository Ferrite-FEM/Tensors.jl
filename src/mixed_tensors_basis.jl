using Tensors 

abstract type TensorBase{dim} end
struct StandardBase{dim} <: TensorBase{dim} end
struct CovariantBase{dim} <: TensorBase{dim} end
struct ContravariantBase{dim} <: TensorBase{dim} end

struct Basis{order, B<:Tuple{Vararg{TensorBase,order}}}
    bases::B
end
getorder(::Basis{order}) where order = order
getbase(b::Basis, i) = b.bases[i]

# Construct from just the type information
function Basis{order, B}() where {order, B<:Tuple}
    return Basis{order, B}(ntuple(i->B.types[i](), length(fieldnames(B))))
end
const StandardBasis{order, dim} = Basis{order, NTuple{order, StandardBase{dim}}}

struct TTensor{B<:Basis, T, N}
    data::NTuple{N,T}
end
getbase(::TTensor{B}, i) where B = getbase(B(), i)
getbase(::TTensor{B}) where B = B()
getorder(::TTensor{B}) where B = getorder(B())

n_components(::Type{<:TensorBase{dim}}) where dim = dim
n_components(tb::Basis) = prod(n_components, tb.bases)

# Can two bases be contracted ? 
contractable(::TensorBase{dim}, ::TensorBase{dim}) where dim = true 
contractable(::TensorBase{d1}, ::TensorBase{d2}) where {d1, d2} = false
contractable(::CovariantBase{dim}, ::CovariantBase{dim}) where dim = false
contractable(::ContravariantBase{dim}, ::ContravariantBase{dim}) where dim = false

function contractable(::typeof(dcontract), S1::TTensor, S2::TTensor)
    (getorder(S1) >= 2 && getorder(S2) >= 2) || return false
    contractable(getbase(S1, getorder(S1)), getbase(S2, 1)) || return false 
    contractable(getbase(S1, getorder(S1)-1), getbase(S2, 2)) || return false
    return true
end

function contractable(::typeof(dot), S1::TTensor, S2::TTensor)
    return contractable(getbase(S1, getorder(S1)), getbase(S2, 1))
end

# Can two bases be added (or subtracted)? 
addable(S1::TTensor, S2::Tensor) = (getbase(S1) == getbase(S2))


getreturnbase(::typeof(otimes), b1::Basis, b2::Basis) = Basis((b1.bases..., b2.bases...))
getreturnbase(::typeof(dot), ::Basis{1}, ::Basis{1}) = nothing # How to handle, will be scalar. Can have Basis{0}?
getreturnbase(::typeof(dot), ::Basis, ::Basis) = Basis(b1.bases[1:end-1]..., b2.bases[2:end]...)