function extract_components{dim, T}(t::Vec{dim, T})
    m = zeros(dim)
    for i in eachindex(t)
        m[i] = t[i]
    end
    return m
end

function extract_components{dim, T}(t::SecondOrderTensor{dim, T})
    m = zeros(dim, dim)
    for i in eachindex(t)
        m[i] = t[i]
    end
    return m
end

function extract_components{dim, T}(t::FourthOrderTensor{dim, T})
    m = zeros(dim, dim, dim, dim)
    for i in eachindex(t)
        m[i] = t[i]
    end
    return m
end


function load_components!(t::Tensors, arr::AbstractArray)
    @assert length(t) == length(arr)
    for i in eachindex(t, arr)
        t[i] = arr[i]
    end
    return t
end


#@gen_code function load_components!{dim, T}(t::SymmetricTensor{2, dim}, mat::Matrix{T})
#    @code :(@assert size(mat) == (dim, dim))
#    @code :(data = get_data(t))
#    k = 1
#    for i in 1:dim, j in 1:i
#        if i == j
#            @code :(@inbounds data[$k] = mat[$i,$j])
#        else
#            @code :(@inbounds data[$k] = 0.5 * (mat[$i,$j] + mat[$j,$i]))
#        end
#        k += 1
#    end
#    @code :(return S)
#end


