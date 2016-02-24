function extract_components{dim, T}(t::AllTensors{dim ,T})
    m = zeros(size(t))
    @inbounds for i in eachindex(t)
        m[i] = t[i]
    end
    return m
end

function load_components!{order, T}(t::AbstractTensor{order}, arr::AbstractArray{T, order})
    @assert size(arr) == size(t)
    @inbounds for i in eachindex(t, arr)
        t[i] = arr[i]
    end
    return t
end


function load_components!{dim, T}(t::SymmetricTensor{2, dim}, arr::Matrix{T})
    @assert size(arr) == size(t)
    @inbounds for i in 1:dim, j in 1:j
        t[i,j] = 0.5 * (arr[i,j] + arr[j,i])
    end
end


