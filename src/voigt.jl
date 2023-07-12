const DEFAULT_VOIGT_ORDER = ([1], [1 3; 4 2], [1 6 5; 9 2 4; 8 7 3])
"""
    tovoigt(A::Union{SecondOrderTensor, FourthOrderTensor}; kwargs...)

Converts a tensor to "Voigt"-format.

Keyword arguments:
 - `offdiagscale`: determines the scaling factor for the offdiagonal elements. 
   This argument is only applicable for `SymmetricTensor`s. `tomandel` can also 
   be used for the "Mandel"-format which sets `offdiagscale = √2` for `SymmetricTensor`s,
   and is equivalent to `tovoigt` for `Tensor`s.
 - `order`: matrix of the linear indices determining the Voigt order. The default
   index order is `[11, 22, 33, 23, 13, 12, 32, 31, 21]`, corresponding to
   `order = [1 6 5; 9 2 4; 8 7 3]`.

See also [`tovoigt!`](@ref) and [`fromvoigt`](@ref).

```jldoctest
julia> tovoigt(Tensor{2,3}(1:9))
9-element Vector{Int64}:
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
6-element Vector{Int64}:
  1
  4
  6
 10
  6
  4

julia> tovoigt(Tensor{4,2}(1:16))
4×4 Matrix{Int64}:
 1  13   9  5
 4  16  12  8
 3  15  11  7
 2  14  10  6
```
"""
function tovoigt end
@inline function tovoigt(A::Tensor{2, dim, T, M}; order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T, M}
    @inbounds tovoigt!(Vector{T}(undef, M), A; order=order)
end
@inline function tovoigt(A::Tensor{4, dim, T, M}; order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T, M}
    @inbounds tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A; order=order)
end
@inline function tovoigt(A::SymmetricTensor{2, dim, T, M}; offdiagscale=one(T), order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T, M}
    @inbounds tovoigt!(Vector{T}(undef, M), A; offdiagscale=offdiagscale, order=order)
end
@inline function tovoigt(A::SymmetricTensor{4, dim, T, M}; offdiagscale=one(T), order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T, M}
    @inbounds tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A; offdiagscale=offdiagscale, order=order)
end

"""
    tovoigt!(v::Array, A::Union{SecondOrderTensor, FourthOrderTensor}; kwargs...)

Converts a tensor to "Voigt"-format using the following index order:
`[11, 22, 33, 23, 13, 12, 32, 31, 21]`.

Keyword arguments:
 - `offset`: offset index for where in the array `A` the tensor should be stored.
   For 4th order tensors the keyword arguments are `offset_i` and `offset_j`,
   respectively. Defaults to `0`.
 - `offdiagscale`: determines the scaling factor for the offdiagonal elements. 
   This argument is only applicable for `SymmetricTensor`s. `frommandel!` can also 
   be used for the "Mandel"-format which sets `offdiagscale = √2` for `SymmetricTensor`s,
   and is equivalent to `fromvoigt!` for `Tensor`s.
 - `order`: matrix of the linear indices determining the Voigt order. The default
   index order is `[11, 22, 33, 23, 13, 12, 32, 31, 21]`.

See also [`tovoigt`](@ref) and [`fromvoigt`](@ref).

```jldoctest
julia> T = rand(Tensor{2,2})
2×2 Tensor{2, 2, Float64, 4}:
 0.590845  0.566237
 0.766797  0.460085

julia> x = zeros(4);

julia> tovoigt!(x, T)
4-element Vector{Float64}:
 0.5908446386657102
 0.4600853424625171
 0.5662374165061859
 0.7667970365022592

julia> x = zeros(5);

julia> tovoigt!(x, T; offset=1)
5-element Vector{Float64}:
 0.0
 0.5908446386657102
 0.4600853424625171
 0.5662374165061859
 0.7667970365022592
```
"""
function tovoigt! end
Base.@propagate_inbounds function tovoigt!(v::AbstractVector, A::Tensor{2, dim}; offset::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim}
    for j in 1:dim, i in 1:dim
        v[offset + order[i, j]] = A[i, j]
    end
    return v
end
Base.@propagate_inbounds function tovoigt!(v::AbstractMatrix, A::Tensor{4, dim}; offset_i::Int=0, offset_j::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim}
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        v[offset_i + order[i, j], offset_j + order[k, l]] = A[i, j, k, l]
    end
    return v
end
Base.@propagate_inbounds function tovoigt!(v::AbstractVector{T}, A::SymmetricTensor{2, dim}; offdiagscale=one(T), offset::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {T, dim}
    for j in 1:dim, i in 1:j
        v[offset + order[i, j]] = i == j ? A[i, j] : A[i, j] * offdiagscale
    end
    return v
end
Base.@propagate_inbounds function tovoigt!(v::AbstractMatrix{T}, A::SymmetricTensor{4, dim}; offdiagscale=one(T), offset_i::Int=0, offset_j::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {T, dim}
    for l in 1:dim, k in 1:l, j in 1:dim, i in 1:j
        v[offset_i + order[i, j], offset_j + order[k, l]] =
            (i == j && k == l) ? A[i, j, k, l] :
            (i == j || k == l) ? A[i, j, k, l] * offdiagscale :
                                 A[i, j, k, l] * (offdiagscale * offdiagscale)
    end
    return v
end

"""
    tomandel(A; kwargs...)

Convert the tensor `A` to voigt-form using the Mandel convention, see [`tovoigt`](@ref).
"""
@inline tomandel(A::SymmetricTensor{o, dim, T}; kwargs...) where{o,dim,T} = tovoigt(A; offdiagscale=√(2one(T)), kwargs...)
@inline tomandel(A::Tensor; kwargs...) = tovoigt(A; kwargs...)

"""
    tomandel!(v, A; kwargs...)

Fill the array `v` with the values in `A` on voigt-form using the Mandel convention, see [`tovoigt!`](@ref).
"""
Base.@propagate_inbounds tomandel!(v::AbstractVecOrMat{T}, A::SymmetricTensor; kwargs...) where{T} = tovoigt!(v, A; offdiagscale=√(2one(T)), kwargs...)
Base.@propagate_inbounds tomandel!(v::AbstractVecOrMat, A::Tensor; kwargs...) = tovoigt!(v, A; kwargs...)

"""
    fromvoigt(S::Type{<:AbstractTensor}, A::Array{T}; kwargs...)

Converts an array `A` stored in Voigt format to a Tensor of type `S`.

Keyword arguments:
 - `offset`: offset index for where in the array `A` the tensor starts. For 4th order
   tensors the keyword arguments are `offset_i` and `offset_j`, respectively.
   Defaults to `0`.
 - `offdiagscale`: determines the scaling factor for the offdiagonal elements. 
   This argument is only applicable for `SymmetricTensor`s. `frommandel` can also 
   be used for the "Mandel"-format which sets `offdiagscale = √2` for `SymmetricTensor`s,
   and is equivalent to `fromvoigt` for `Tensor`s.
 - `order`: matrix of the linear indices determining the Voigt order. The default
   index order is `[11, 22, 33, 23, 13, 12, 32, 31, 21]`, corresponding to
   `order = [1 6 5; 9 2 4; 8 7 3]`.

See also [`tovoigt`](@ref).

```jldoctest
julia> fromvoigt(Tensor{2,3}, 1.0:1.0:9.0)
3×3 Tensor{2, 3, Float64, 9}:
 1.0  6.0  5.0
 9.0  2.0  4.0
 8.0  7.0  3.0
```
"""
Base.@propagate_inbounds function fromvoigt(TT::Type{<: Tensor{2, dim}}, v::AbstractVector; offset::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim}
    return TT(function (i, j); return v[offset + order[i, j]]; end)
end
Base.@propagate_inbounds function fromvoigt(TT::Type{<: Tensor{4, dim}}, v::AbstractMatrix; offset_i::Int=0, offset_j::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim}
    return TT(function (i, j, k, l); return v[offset_i + order[i, j], offset_j + order[k, l]]; end)
end
Base.@propagate_inbounds function fromvoigt(TT::Type{<: SymmetricTensor{2, dim}}, v::AbstractVector{T}; offdiagscale = one(T), offset::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T}
    return TT(function (i, j)
            i > j && ((i, j) = (j, i))
            i == j ? (return v[offset + order[i, j]]) :
                     (return T(v[offset + order[i, j]] / offdiagscale))
        end)
end
Base.@propagate_inbounds function fromvoigt(TT::Type{<: SymmetricTensor{4, dim}}, v::AbstractMatrix{T}; offdiagscale = one(T), offset_i::Int=0, offset_j::Int=0, order=DEFAULT_VOIGT_ORDER[dim]) where {dim, T}
    return TT(function (i, j, k, l)
            i > j && ((i, j) = (j, i))
            k > l && ((k, l) = (l, k))
            i == j && k == l ? (return v[offset_i + order[i, j], offset_j + order[k, l]]) :
            i == j || k == l ? (return T(v[offset_i + order[i, j], offset_j + order[k, l]] / offdiagscale)) :
                               (return T(v[offset_i + order[i, j], offset_j + order[k, l]] / (offdiagscale * offdiagscale)))
        end)
end

"""
    frommandel(TT, v; kwargs...)

Convert the Array `v` in voigt-format, following the Mandel convention, to a tensor of type `TT`, see [`fromvoigt`](@ref).
"""
Base.@propagate_inbounds frommandel(TT::Type{<: SymmetricTensor}, v::AbstractVecOrMat{T}; kwargs...) where{T} = fromvoigt(TT, v; offdiagscale=√(2one(T)), kwargs...)
Base.@propagate_inbounds frommandel(TT::Type{<: Tensor}, v::AbstractVecOrMat; kwargs...) = fromvoigt(TT, v; kwargs...)
