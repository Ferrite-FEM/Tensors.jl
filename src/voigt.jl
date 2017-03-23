dimrange(::Tensor,          i, dim) = dim
dimrange(::SymmetricTensor, i, dim) = i

"""
    tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; offdiagscale)

Converts a tensor to "Voigt"-format using the following index order:
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`.
For `SymmetricTensor`s, the keyword argument `offdiagscale` sets a scaling factor
on the offdiagonal elements

See also [`fromvoigt`](@ref).

```jldoctest
julia> tovoigt(Tensor{2,3}(1:9))
9-element Array{Int64,1}:
 1
 5
 9
 8
 7
 4
 6
 3
 2

julia> tovoigt(SymmetricTensor{2,3}(1.0:1.0:6.0); offdiagscale = 2.0)
6-element Array{Float64,1}:
  1.0
  4.0
  6.0
 10.0
  6.0
  4.0

julia> tovoigt(Tensor{4,2}(1:16))
4×4 Array{Int64,2}:
 1  13   9  5
 4  16  12  8
 3  15  11  7
 2  14  10  6
```

"""
function tovoigt end

tovoigt(A::SymmetricTensor; offdiagscale::Real=one(eltype(A))) = _tovoigt(A, offdiagscale)
tovoigt(A::Union{Tensor{2}, Tensor{4}}) = _tovoigt(A, one(eltype(A)))

function _tovoigt{dim,T}(A::SecondOrderTensor{dim, T}, offdiagscale)
    v = zeros(T, length(A.data))
    for j in 1:dim, i in 1:dimrange(A, j, dim)
        I, si = _index_scale(dim, i, j, offdiagscale)
        v[I] = si * A[i,j]
    end
    return v
end

function _tovoigt{dim,T}(A::FourthOrderTensor{dim, T}, offdiagscale)
    n = Int(√(length(A.data)))
    v = zeros(T, n, n)
     for l in 1:dim, k in 1:dimrange(A, l, dim), j in 1:dim, i in 1:dimrange(A, j, dim)
        I, si = _index_scale(dim, i, j, offdiagscale)
        J, sj = _index_scale(dim, k, l, offdiagscale)
        v[I,J] = si * sj * A[i,j,k,l]
    end
    return v
end

"""
    fromvoigt(::T, A::Array)

Converts an array `A` stored in Voigt format to a Tensor of type `T`.
For `SymmetricTensor`s, the keyword argument `offdiagscale` sets an inverse scaling factor
on the offdiagonal elements.

See also [`tovoigt`](@ref).

```jldoctest
julia> fromvoigt(Tensor{2,3}, 1.0:1.0:9.0)
3×3 Tensors.Tensor{2,3,Float64,9}:
 1.0  6.0  5.0
 9.0  2.0  4.0
 8.0  7.0  3.0
```
"""
function fromvoigt end

fromvoigt{dim}(A::Union{Type{SymmetricTensor{2,dim}}, Type{SymmetricTensor{4,dim}}}, v::AbstractVecOrMat; offdiagscale::Real=one(eltype(v))) = _fromvoigt(A, v, offdiagscale)
fromvoigt{dim}(A::Union{Type{Tensor{2,dim}}, Type{Tensor{4,dim}}}, v::AbstractVecOrMat) = _fromvoigt(A, v, one(eltype(v)))

function _fromvoigt{dim}(TT::Union{Type{Tensor{2,dim}}, Type{SymmetricTensor{2,dim}}}, v::AbstractVector, offdiagscale)
    length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i,j)
            if i > j && TT == SymmetricTensor{2,dim}
                i,j = j,i
            end
            I, si = _index_scale(dim, i, j, offdiagscale)
            return v[I] / si
        end
    )
end

function _fromvoigt{dim}(TT::Union{Type{Tensor{4,dim}}, Type{SymmetricTensor{4,dim}}}, v::AbstractMatrix, offdiagscale)
    length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i,j,k,l)
            if TT == SymmetricTensor{4,dim}
                if i > j; i,j = j,i; end
                if k > l; k,l = l,k; end
            end
            I, si = _index_scale(dim, i, j, offdiagscale)
            J, sj = _index_scale(dim, k, l, offdiagscale)
            return v[I, J] / (si * sj)
        end
    )
end

# Get index and scale to reduce order of symmetric tensor
#
# Example
# index:
# [1 6 5
#  9 2 4
#  8 7 3]
#
# scale:
# [1 s s
#  s 1 s
#  s s 1]
#
function _index_scale(dim::Int, i::Int, j::Int, s::Real) # i ≤ dim and j ≤ dim are assumed
    if i == j
        (i, one(typeof(s)))
    elseif i < j
        (_offdiagind(dim, i, j), s)
    else
        (_offdiagind(dim, j, i) + sum(1:dim-1), s)
    end
end

function _offdiagind(dim::Int, i::Int, j::Int) # i < j ≤ dim is assumed
    count = dim + (j-1) - (i-1)
    for idx in j+1:dim
        count += idx-1
    end
    count
end