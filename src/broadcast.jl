#################
# Broadcasting  #
#################
# Broadcasts where the only container arguments are tensors of one shape return
# a tensor instead of silently materializing an `Array` (issue #223):
#
#   * tensor ∘ scalar         -> tensor (dense; a SymmetricTensor densifies,
#                                since a general `f` need not preserve symmetry)
#   * tensor ∘ tensor         -> tensor, when both have the same dense base
#   * tensor ∘ ordinary Array -> Array (usual broadcast semantics)
#   * non-Number result       -> Array (tensors only hold Numbers)

import Base.Broadcast: BroadcastStyle, Broadcasted, DefaultArrayStyle

struct TensorStyle{TB} <: BroadcastStyle end # TB = dense base type, e.g. Tensor{2,3}

dense_base(::Type{<:Tensor{order, dim}}) where {order, dim} = Tensor{order, dim}
dense_base(::Type{<:SymmetricTensor{order, dim}}) where {order, dim} = Tensor{order, dim}
dense_base(::Type{<:MixedTensor{order, dims}}) where {order, dims} = MixedTensor{order, dims}

base_ndims(::Type{<:Tensor{order}}) where {order} = order
base_ndims(::Type{<:MixedTensor{order}}) where {order} = order

BroadcastStyle(::Type{TT}) where {TT <: AbstractTensor} = TensorStyle{dense_base(TT)}()
BroadcastStyle(style::TensorStyle{TB}, ::TensorStyle{TB}) where {TB} = style
# tensors of different shapes: ordinary array broadcasting
BroadcastStyle(::TensorStyle{TB1}, ::TensorStyle{TB2}) where {TB1, TB2} =
    DefaultArrayStyle{max(base_ndims(TB1), base_ndims(TB2))}()
# with scalars: stay a tensor; with ordinary arrays: fall back to Array
BroadcastStyle(style::TensorStyle, ::DefaultArrayStyle{0}) = style
BroadcastStyle(::TensorStyle{TB}, ::DefaultArrayStyle{N}) where {TB, N} =
    DefaultArrayStyle{max(base_ndims(TB), N)}()

@inline _bc_el(a::AbstractTensor, inds) = @inbounds a[inds...]
@inline _bc_el(a::Ref, inds) = a[]
@inline _bc_el(a, inds) = a

@inline function Base.copy(bc::Broadcasted{TensorStyle{TB}}) where {TB}
    bcf = Broadcast.flatten(bc)
    ET = Broadcast.combine_eltypes(bcf.f, bcf.args)
    if !(ET <: Number)
        # tensors hold Numbers only; materialize as an Array
        return copy(Broadcasted{DefaultArrayStyle{base_ndims(TB)}}(bcf.f, bcf.args))
    end
    f = bcf.f
    args = bcf.args
    return TB(@inline function(inds...)
        f(map(Base.Fix2(_bc_el, inds), args)...)
    end)
end
