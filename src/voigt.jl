const VOIGT_ORDER = ([1], [1 3; 4 2], [1 6 5; 9 2 4; 8 7 3])
"""
    tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; offdiagscale)

Converts a tensor to "Voigt"-format using the following index order:
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`.
For `SymmetricTensor`s, the keyword argument `offdiagscale` sets a scaling factor
on the offdiagonal elements. `tomandel` can also be used for "Mandel"-format
which sets `offdiagscale = √2`.

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
tovoigt{dim, T, M}(A::Tensor{2, dim, T, M})                            =  tovoigt!(Vector{T}(M), A)
tovoigt{dim, T, M}(A::Tensor{4, dim, T, M})                            =  tovoigt!(Matrix{T}(Int(√M), Int(√M)), A)
tovoigt{dim, T, M}(A::SymmetricTensor{2, dim, T, M}; offdiagscale = 1) = _tovoigt!(Vector{T}(M), A, offdiagscale)
tovoigt{dim, T, M}(A::SymmetricTensor{4, dim, T, M}; offdiagscale = 1) = _tovoigt!(Matrix{T}(Int(√M), Int(√M)), A, offdiagscale)

function tovoigt!{dim}(v::AbstractVector, A::Tensor{2, dim})
    length(v) == length(A.data) || throw(ArgumentError("invalid input size of voigt array"))
    @inbounds for j in 1:dim, i in 1:dim
        v[VOIGT_ORDER[dim][i, j]] = A[i, j]
    end
    v
end
function tovoigt!{dim}(v::AbstractMatrix, A::Tensor{4, dim})
    (size(v, 1) == size(v, 2) && length(v) == length(A.data)) || throw(ArgumentError("invalid input size of voigt array"))
    @inbounds for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] = A[i, j, k, l]
    end
    v
end
tovoigt!(v::AbstractVecOrMat, A::SymmetricTensor; offdiagscale = 1) = _tovoigt!(v, A, offdiagscale)
function _tovoigt!{dim}(v::AbstractVector, A::SymmetricTensor{2, dim}, offdiagscale)
    length(v) == length(A.data) || throw(ArgumentError("invalid input size of voigt array"))
    @inbounds for j in 1:dim, i in 1:j
        v[VOIGT_ORDER[dim][i, j]] = i == j ? A[i, j] : A[i, j] * offdiagscale
    end
    v
end
function _tovoigt!{dim}(v::AbstractMatrix, A::SymmetricTensor{4, dim}, offdiagscale)
    (size(v, 1) == size(v, 2) && length(v) == length(A.data)) || throw(ArgumentError("invalid input size of voigt array"))
    @inbounds for l in 1:dim, k in 1:l, j in 1:dim, i in 1:j
        v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] =
            (i == j && k == l) ? A[i, j, k, l] :
            (i == j || k == l) ? A[i, j, k, l] * offdiagscale :
                                 A[i, j, k, l] * (offdiagscale * offdiagscale)
    end
    v
end

tomandel(A::SymmetricTensor) = tovoigt(A, offdiagscale = √2)
tomandel!(v::AbstractVecOrMat, A::SymmetricTensor) = _tovoigt!(v, A, √2)

"""
    fromvoigt(T::Type{<:AbstractTensor}, A::Array)

Converts an array `A` stored in Voigt format to a Tensor of type `T`.
For `SymmetricTensor`s, the keyword argument `offdiagscale` sets an inverse scaling factor
on the offdiagonal elements. `frommandel` can also be used for "Mandel"-format
which sets `offdiagscale = √2`.

See also [`tovoigt`](@ref).

```jldoctest
julia> fromvoigt(Tensor{2,3}, 1.0:1.0:9.0)
3×3 Tensors.Tensor{2,3,Float64,9}:
 1.0  6.0  5.0
 9.0  2.0  4.0
 8.0  7.0  3.0
```
"""
function fromvoigt{dim, T}(TT::Type{Tensor{2, dim}}, v::AbstractVector{T})
    length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i, j); @inboundsret T(v[VOIGT_ORDER[dim][i, j]]); end)
end
function fromvoigt{dim, T}(TT::Type{Tensor{4, dim}}, v::AbstractMatrix{T})
    size(v, 1) == size(v, 2) && length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i, j, k, l); @inboundsret T(v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]]); end)
end
fromvoigt{dim}(A::Union{Type{SymmetricTensor{2, dim}}, Type{SymmetricTensor{4, dim}}}, v::AbstractVecOrMat; offdiagscale = 1) = _fromvoigt(A, v, offdiagscale)
function _fromvoigt{dim, T}(TT::Type{SymmetricTensor{2, dim}}, v::AbstractVector{T}, offdiagscale)
    length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i, j)
            i > j && ((i, j) = (j, i))
            i == j ? (@inboundsret T(v[VOIGT_ORDER[dim][i, j]])) :
                     (@inboundsret T(v[VOIGT_ORDER[dim][i, j]] / offdiagscale))
        end)
end
function _fromvoigt{dim, T}(TT::Type{SymmetricTensor{4, dim}}, v::AbstractMatrix{T}, offdiagscale)
    size(v, 1) == size(v, 2) && length(v) == n_components(TT) || throw(ArgumentError("invalid input size of voigt array"))
    return TT(function (i, j, k, l)
            i > j && ((i, j) = (j, i))
            k > l && ((k, l) = (l, k))
            i == j && k == l ? (@inboundsret T(v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]])) :
            i == j || k == l ? (@inboundsret T(v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] / offdiagscale)) :
                               (@inboundsret T(v[VOIGT_ORDER[dim][i, j], VOIGT_ORDER[dim][k, l]] / (offdiagscale * offdiagscale)))
        end)
end

frommandel{dim, T}(A::Union{Type{SymmetricTensor{2, dim}}, Type{SymmetricTensor{4, dim}}}, v::AbstractVecOrMat{T}) = _fromvoigt(A, v, T(√2))
