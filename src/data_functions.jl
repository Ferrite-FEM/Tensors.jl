# Handles getting data out of tensors into e.g vector
function extract_components{dim, T}(t::AllTensors{dim ,T})
    m = zeros(size(t))
    @inbounds for i in eachindex(t)
        m[i] = t[i]
    end
    return m
end

function store!(v::Vector, t::AllTensors, offset = 0)
    for (I,i) in enumerate(offset+1:offset+length(t))
        v[i] = t[I]
    end
    return v
end

