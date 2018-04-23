const VOIGT_ORDER = ([0], [0 2; 3 1], [0 5 4; 8 1 3; 7 6 2])
"""
    tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; offdiagscale)
    tovoigt!(v::Array, A::Union{SecondOrderTensor, FourthOrderTensor}; offdiagscale)

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

julia> tovoigt(SymmetricTensor{2,3}(1:6); offdiagscale = 2)
6-element Array{Int64,1}:
  1
  4
  6
 10
  6
  4

julia> tovoigt(Tensor{4,2}(1:16))
4×4 Array{Int64,2}:
 1  13   9  5
 4  16  12  8
 3  15  11  7
 2  14  10  6
```
"""
@inline function tovoigt(A::Tensor{2, dim, T, M}) where {dim, T, M}
    @inbounds tovoigt!(Vector{T}(undef, M), A)
end
@inline function tovoigt(A::Tensor{4, dim, T, M}) where {dim, T, M}
    @inbounds tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A)
end
@inline function tovoigt(A::SymmetricTensor{2, dim, T, M}; offdiagscale::T = one(T)) where {dim, T, M}
    @inbounds tovoigt!(Vector{T}(undef, M), A, offdiagscale = offdiagscale)
end
@inline function tovoigt(A::SymmetricTensor{4, dim, T, M}; offdiagscale::T = one(T)) where {dim, T, M}
    @inbounds tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A, offdiagscale = offdiagscale)
end

Base.@propagate_inbounds @inline function tovoigt!(v::AbstractVector, A::Tensor{2, dim}; offset::Int = 1) where {dim}
    for j in 1:dim, i in 1:dim
        v[offset + VOIGT_ORDER[dim][i, j]] = A[i, j]
    end
    return v
end
Base.@propagate_inbounds @inline function tovoigt!(v::AbstractMatrix, A::Tensor{4, dim}; offset_i::Int = 1, offset_j::Int = 1) where {dim}
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]] = A[i, j, k, l]
    end
    return v
end
Base.@propagate_inbounds @inline function tovoigt!(v::AbstractVector{T}, A::SymmetricTensor{2, dim}; offdiagscale::T = one(T), offset::Int = 1) where {T, dim}
    for j in 1:dim, i in 1:j
        v[offset + VOIGT_ORDER[dim][i, j]] = i == j ? A[i, j] : A[i, j] * offdiagscale
    end
    return v
end
Base.@propagate_inbounds @inline function tovoigt!(v::AbstractMatrix{T}, A::SymmetricTensor{4, dim}; offdiagscale::T = one(T), offset_i::Int = 1, offset_j::Int = 1) where {T, dim}
    for l in 1:dim, k in 1:l, j in 1:dim, i in 1:j
        v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]] =
            (i == j && k == l) ? A[i, j, k, l] :
            (i == j || k == l) ? A[i, j, k, l] * offdiagscale :
                                 A[i, j, k, l] * (offdiagscale * offdiagscale)
    end
    return v
end

@inline tomandel(A::SymmetricTensor{order, dim, T}) where {order, dim, T} = @inbounds tovoigt(A, offdiagscale = T(√2))
Base.@propagate_inbounds @inline function tomandel!(v::AbstractVector{T}, A::SymmetricTensor{2}; offset::Int = 1) where {T}
    tovoigt!(v, A, offdiagscale = T(√2), offset = offset)
end
Base.@propagate_inbounds @inline function tomandel!(v::AbstractMatrix{T}, A::SymmetricTensor{4}; offset_i::Int = 1, offset_j::Int = 1) where {T}
    tovoigt!(v, A, offdiagscale = T(√2), offset_i = offset_i, offset_j = offset_j)
end

"""
    fromvoigt(S::Type{<:AbstractTensor}, A::Array{T}; offdiagscale::T = 1)

Converts an array `A` stored in Voigt format to a Tensor of type `S`.
For `SymmetricTensor`s, the keyword argument `offdiagscale` sets an inverse scaling factor
on the offdiagonal elements. `frommandel` can also be used for "Mandel"-format
which sets `offdiagscale = √2`.

See also [`tovoigt`](@ref).

```jldoctest
julia> fromvoigt(Tensor{2,3}, 1.0:1.0:9.0)
3×3 Tensor{2,3,Float64,9}:
 1.0  6.0  5.0
 9.0  2.0  4.0
 8.0  7.0  3.0
```
"""
Base.@propagate_inbounds @inline function fromvoigt(TT::Type{<: Tensor{2, dim}}, v::AbstractVector; offset::Int = 1) where {dim}
    return TT(function (i, j); return v[offset + VOIGT_ORDER[dim][i, j]]; end)
end
Base.@propagate_inbounds @inline function fromvoigt(TT::Type{<: Tensor{4, dim}}, v::AbstractMatrix; offset_i::Int = 1, offset_j::Int = 1) where {dim}
    return TT(function (i, j, k, l); return v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]]; end)
end
Base.@propagate_inbounds @inline function fromvoigt(TT::Type{<: SymmetricTensor{2, dim}}, v::AbstractVector{T}; offdiagscale::T = T(1), offset::Int = 1) where {dim, T}
    return TT(function (i, j)
            i > j && ((i, j) = (j, i))
            i == j ? (return v[offset + VOIGT_ORDER[dim][i, j]]) :
                     (return v[offset + VOIGT_ORDER[dim][i, j]] / offdiagscale)
        end)
end
Base.@propagate_inbounds @inline function fromvoigt(TT::Type{<: SymmetricTensor{4, dim}}, v::AbstractMatrix{T}; offdiagscale::T = T(1), offset_i::Int = 1, offset_j::Int = 1) where {dim, T}
    return TT(function (i, j, k, l)
            i > j && ((i, j) = (j, i))
            k > l && ((k, l) = (l, k))
            i == j && k == l ? (return v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]]) :
            i == j || k == l ? (return v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]] / offdiagscale) :
                               (return v[offset_i + VOIGT_ORDER[dim][i, j], offset_j + VOIGT_ORDER[dim][k, l]] / (offdiagscale * offdiagscale))
        end)
end

Base.@propagate_inbounds @inline function frommandel(TT::Type{<: SymmetricTensor{2}}, v::AbstractVector{T}; offset::Int = 1) where {T}
    fromvoigt(TT, v, offdiagscale = T(√2), offset = offset)
end
Base.@propagate_inbounds @inline function frommandel(TT::Type{<: SymmetricTensor{4}}, v::AbstractMatrix{T}; offset_i::Int = 1, offset_j::Int = 1) where {T}
    fromvoigt(TT, v, offdiagscale = T(√2), offset_i = offset_i, offset_j = offset_j)
end
