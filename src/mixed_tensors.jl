# Convert to regular tensor if possible
# isregular required for type stability
isregular(::Type{<:MixedTensor{1}}) = true
isregular(::Type{<:MixedTensor{2, dims}}) where {dims} = dims[1] === dims[2]
isregular(::Type{<:MixedTensor{3, dims}}) where {dims} = dims[1] === dims[2] === dims[3]
isregular(::Type{<:MixedTensor{4, dims}}) where {dims} = dims[1] === dims[2] === dims[3] === dims[4]

function regular_if_possible(::Type{TT}) where {order, dims, TT <: MixedTensor{order, dims}}
    return isregular(TT) ? Tensor{order, dims[1]} : TT
end
function regular_if_possible(::Type{TT}) where {order, dims, T, TT <: MixedTensor{order, dims, T}}
    return isregular(TT) ? Tensor{order, dims[1], T} : TT
end
function regular_if_possible(::Type{TT}) where {order, dims, T, M, TT <: MixedTensor{order, dims, T, M}}
    return isregular(TT) ? Tensor{order, dims[1], T, M} : TT
end
function regular_if_possible(t::TT) where {order, dims, TT <: MixedTensor{order, dims}}
    return isregular(TT) ? Tensor{order, dims[1]}(get_data(t)) : t
end

makemixed(t::Tensor{1, dim}) where {dim} = MixedTensor{1, (dim,)}(get_data(t))
makemixed(t::Tensor{2, dim}) where {dim} = MixedTensor{2, (dim, dim)}(get_data(t))
makemixed(t::Tensor{3, dim}) where {dim} = MixedTensor{3, (dim, dim, dim)}(get_data(t))
makemixed(t::Tensor{4, dim}) where {dim} = MixedTensor{4, (dim, dim, dim, dim)}(get_data(t))
makemixed(t::MixedTensor) = t
