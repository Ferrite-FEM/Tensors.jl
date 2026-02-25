# Convert to regular tensor if possible
# isregular required for type stability
isregular(::Type{<:MixedTensor{1}}) = true
isregular(::Type{<:MixedTensor2{dim, dim}}) where {dim} = true
isregular(::Type{<:MixedTensor3{dim, dim, dim}}) where {dim} = true
isregular(::Type{<:MixedTensor4{dim, dim, dim, dim}}) where {dim} = true
isregular(::Type{<:MixedTensor}) = false

function regular_if_possible(::Type{TT}) where {order, dims, TT <: MixedTensor{order, dims}}
    return isregular(TT) ? Tensor{order, size(TT)[1]} : TT
end
function regular_if_possible(::Type{TT}) where {order, dims, T, TT <: MixedTensor{order, dims, T}}
    return isregular(TT) ? Tensor{order, size(TT)[1], T} : TT
end
function regular_if_possible(::Type{TT}) where {order, dims, T, M, TT <: MixedTensor{order, dims, T, M}}
    return isregular(TT) ? Tensor{order, size(TT)[1], T, M} : TT
end
function regular_if_possible(t::TT) where {order, dims, TT <: MixedTensor{order, dims}}
    return isregular(TT) ? Tensor{order, size(TT)[1]}(get_data(t)) : t
end

makemixed(t::Tensor{1, dim}) where {dim} = MixedTensor{1, Tuple{dim}}(get_data(t))
makemixed(t::Tensor{2, dim}) where {dim} = MixedTensor2{dim, dim}(get_data(t))
makemixed(t::Tensor{3, dim}) where {dim} = MixedTensor3{dim, dim, dim}(get_data(t))
makemixed(t::Tensor{4, dim}) where {dim} = MixedTensor4{dim, dim, dim, dim}(get_data(t))
makemixed(t::MixedTensor) = t
