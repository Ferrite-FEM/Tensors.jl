# Convert to regular tensor if possible
# isregular required for type stability
isregular(::MixedTensor{1}) = true
isregular(::MixedTensor{2, dims}) where {dims} = dims[1] === dims[2]
isregular(::MixedTensor{3, dims}) where {dims} = dims[1] === dims[2] === dims[3]
isregular(::MixedTensor{4, dims}) where {dims} = dims[1] === dims[2] === dims[3] === dims[4]

function makeregular(t::MixedTensor{order,dims}) where {order,dims}
    if isregular(t)
        return Tensor{order,dims[1]}(get_data(t))
    else
        return t
    end
end
makemixed(t::Tensor{1, dim}) where {dim} = MixedTensor{1, (dim,)}(get_data(t))
makemixed(t::Tensor{2, dim}) where {dim} = MixedTensor{2, (dim, dim)}(get_data(t))
makemixed(t::Tensor{3, dim}) where {dim} = MixedTensor{3, (dim, dim, dim)}(get_data(t))
makemixed(t::Tensor{4, dim}) where {dim} = MixedTensor{4, (dim, dim, dim, dim)}(get_data(t))
makemixed(t::MixedTensor) = t
