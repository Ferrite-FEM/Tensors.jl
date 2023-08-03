const DEFAULT_VOIGT_ORDER = ([1], [1 3; 4 2], [1 6 5; 9 2 4; 8 7 3])
"""
    tovoigt([type::Type{<:AbstractArray}, ]A::Union{SecondOrderTensor, FourthOrderTensor}; kwargs...)

Converts a tensor to "Voigt"-format.

Optional argument:
- `type`: determines the returned Array type. Possible types are `Array` and `SArray` (see
  [`StaticArrays`](https://juliaarrays.github.io/StaticArrays.jl/stable/)). 
  
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

julia> tovoigt(SMatrix, Tensor{4,2}(1:16))
4×4 SMatrix{4, 4, Int64, 16} with indices SOneTo(4)×SOneTo(4):
 1  13   9  5
 4  16  12  8
 3  15  11  7
 2  14  10  6
```

"""
function tovoigt end
# default to regular Array
@inline function tovoigt(A::AbstractTensor; kwargs...)
    return tovoigt(Array, A; kwargs...)
end
@inline tovoigt(::Type{Array}, A::SecondOrderTensor; kwargs...) = tovoigt(Vector, A; kwargs...)
@inline tovoigt(::Type{Array}, A::FourthOrderTensor; kwargs...) = tovoigt(Matrix, A; kwargs...)

@inline function tovoigt(::Type{<:Vector}, A::Tensor{2, dim, T, M}; order=nothing) where {dim, T, M}
    @inbounds _tovoigt!(Vector{T}(undef, M), A, order)
end
@inline function tovoigt(::Type{<:Matrix}, A::Tensor{4, dim, T, M}; order=nothing) where {dim, T, M}
    @inbounds _tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A, order)
end
@inline function tovoigt(::Type{<:Vector}, A::SymmetricTensor{2, dim, T, M}; offdiagscale=one(T), order=nothing) where {dim, T, M}
    @inbounds _tovoigt!(Vector{T}(undef, M), A, order; offdiagscale=offdiagscale)
end
@inline function tovoigt(::Type{<:Matrix}, A::SymmetricTensor{4, dim, T, M}; offdiagscale=one(T), order=nothing) where {dim, T, M}
    @inbounds _tovoigt!(Matrix{T}(undef, Int(√M), Int(√M)), A, order; offdiagscale=offdiagscale)
end

"""
    tovoigt!(v::AbstractArray, A::Union{SecondOrderTensor, FourthOrderTensor}; kwargs...)

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
Base.@propagate_inbounds function tovoigt!(v::AbstractVector, A::Tensor{2}; offset::Int=0, order=nothing)
    _tovoigt!(v, A, order; offset=offset)
end
Base.@propagate_inbounds function tovoigt!(v::AbstractMatrix, A::Tensor{4}; offset_i::Int=0, offset_j::Int=0, order=nothing)
    _tovoigt!(v, A, order; offset_i=offset_i, offset_j=offset_j)
end
Base.@propagate_inbounds function tovoigt!(v::AbstractVector{T}, A::SymmetricTensor{2}; offdiagscale=one(T), offset::Int=0, order=nothing) where T
    _tovoigt!(v, A, order; offdiagscale=offdiagscale, offset=offset)
end
Base.@propagate_inbounds function tovoigt!(v::AbstractMatrix{T}, A::SymmetricTensor{4}; offdiagscale=one(T), offset_i::Int=0, offset_j::Int=0, order=nothing) where T
    _tovoigt!(v, A, order; offdiagscale=offdiagscale, offset_i=offset_i, offset_j=offset_j)
end

# default voigt order (faster than custom voigt order)
Base.@propagate_inbounds function _tovoigt!(v::AbstractVecOrMat{T}, A::SecondOrderTensor{dim,T}, ::Nothing; offset=0, offdiagscale=one(T)) where {dim,T}
    tuple_data, = _to_voigt_tuple(A, offdiagscale)
    for i in eachindex(tuple_data)
        v[offset+i] = tuple_data[i]
    end
    return v
end
Base.@propagate_inbounds function _tovoigt!(v::AbstractVecOrMat{T}, A::FourthOrderTensor{dim,T}, ::Nothing; offdiagscale=one(T), offset_i=0, offset_j=0) where {dim,T}
    tuple_data, N = _to_voigt_tuple(A, offdiagscale)
    cartesian = CartesianIndices(((offset_i+1):(offset_i+N), (offset_j+1):(offset_j+N)))
    for i in eachindex(tuple_data)
        v[cartesian[i]] = tuple_data[i]
    end
    return v
end

# custom voigt order (slower than default voigt order)
Base.@propagate_inbounds function _tovoigt!(v::AbstractVector, A::Tensor{2, dim}, order::AbstractVecOrMat; offset::Int=0) where {dim}
    for j in 1:dim, i in 1:dim
        v[offset + order[i, j]] = A[i, j]
    end
    return v
end
Base.@propagate_inbounds function _tovoigt!(v::AbstractMatrix, A::Tensor{4, dim}, order::AbstractVecOrMat; offset_i::Int=0, offset_j::Int=0) where {dim}
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        v[offset_i + order[i, j], offset_j + order[k, l]] = A[i, j, k, l]
    end
    return v
end
Base.@propagate_inbounds function _tovoigt!(v::AbstractVector{T}, A::SymmetricTensor{2, dim}, order::AbstractVecOrMat; offdiagscale=one(T), offset::Int=0) where {T, dim}
    for j in 1:dim, i in 1:j
        v[offset + order[i, j]] = i == j ? A[i, j] : A[i, j] * offdiagscale
    end
    return v
end
Base.@propagate_inbounds function _tovoigt!(v::AbstractMatrix{T}, A::SymmetricTensor{4, dim}, order::AbstractVecOrMat; offdiagscale=one(T), offset_i::Int=0, offset_j::Int=0) where {T, dim}
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
@inline tomandel(A::AbstractTensor; kwargs...) = tomandel(Array, A; kwargs...)
@inline tomandel(::Type{AT}, A::SymmetricTensor{o, dim, T}; kwargs...) where{AT,o,dim,T} = tovoigt(AT, A; offdiagscale=√(2one(T)), kwargs...)
@inline tomandel(::Type{AT}, A::Tensor; kwargs...) where AT = tovoigt(AT, A; kwargs...)

"""
    tomandel!(v, A; kwargs...)

Fill the array `v` with the values in `A` on voigt-form using the Mandel convention, see [`tovoigt!`](@ref).
"""
Base.@propagate_inbounds tomandel!(v::AbstractVecOrMat{T}, A::SymmetricTensor; kwargs...) where{T} = tovoigt!(v, A; offdiagscale=√(2one(T)), kwargs...)
Base.@propagate_inbounds tomandel!(v::AbstractVecOrMat, A::Tensor; kwargs...) = tovoigt!(v, A; kwargs...)

"""
    fromvoigt(S::Type{<:AbstractTensor}, A::AbstractArray{T}; kwargs...)

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

##############################################################
# Reorder tensor data to tuple in default voigt format order #
##############################################################
function __to_voigt_tuple(A::Type{TT}, s=one(T)) where {TT<:SecondOrderTensor{dim,T}} where {dim,T}
    # Define internals for generation
    idx_fun(i, j) = compute_index(get_base(A), i, j)
    maxind(j) = TT<:SymmetricTensor ? j : dim
    N = n_components(get_base(A))

    exps = Expr(:tuple)
    append!(exps.args, [nothing for _ in 1:N]) # "Preallocate" to allow indexing directly

    for j in 1:dim, i in 1:maxind(j)
        voigt_ind = DEFAULT_VOIGT_ORDER[dim][i,j]
        if i==j
            exps.args[voigt_ind] = :(get_data(A)[$(idx_fun(i, j))])
        else
            exps.args[voigt_ind] = :(s*get_data(A)[$(idx_fun(i, j))])
        end
    end
    return exps, N
end

function __to_voigt_tuple(A::Type{TT}, s=one(T)) where {TT<:FourthOrderTensor{dim,T}} where {dim,T}
    # Define internals for generation
    idx_fun(i, j, k, l) = compute_index(get_base(A), i, j, k, l) # why no change needed above?
    maxind(j) = TT<:SymmetricTensor ? j : dim
    voigt_lin_index(vi, vj) = (vj-1)*N + vi 
    N = Int(sqrt(n_components(get_base(A))))
    
    exps = Expr(:tuple)
    append!(exps.args, [nothing for _ in 1:N^2]) # "Preallocate" to allow indexing directly

    for l in 1:dim, k in 1:maxind(l), j in 1:dim, i in 1:maxind(j)
        voigt_lin_ind = voigt_lin_index(DEFAULT_VOIGT_ORDER[dim][i,j], DEFAULT_VOIGT_ORDER[dim][k,l])
        if i==j && k==l
            exps.args[voigt_lin_ind] = :(get_data(A)[$(idx_fun(i, j, k, l))])
        elseif i!=j && k!=l
            exps.args[voigt_lin_ind] = :(s*s*get_data(A)[$(idx_fun(i, j, k, l))])
        else
            exps.args[voigt_lin_ind] = :(s*get_data(A)[$(idx_fun(i, j, k, l))])
        end
    end
    return exps, N
end

@generated function _to_voigt_tuple(A::AbstractTensor{order, dim, T}, s=one(T)) where {order,dim,T}
    exps, N = __to_voigt_tuple(A, s)
    quote
        $(Expr(:meta, :inline))
        @inbounds return $exps, $N
    end
end

########################
# StaticArrays support #
########################
@inline tovoigt(::Type{SArray}, A::SecondOrderTensor; kwargs...) = tovoigt(SVector, A; kwargs...)
@inline tovoigt(::Type{SArray}, A::FourthOrderTensor; kwargs...) = tovoigt(SMatrix, A; kwargs...)
@inline function tovoigt(::Type{<:SVector}, A::Tensor{2, dim, T, M}; order=nothing) where {dim, T, M}
    @inbounds _to_static_voigt(A, order)
end
@inline function tovoigt(::Type{<:SMatrix}, A::Tensor{4, dim, T, M}; order=nothing) where {dim, T, M}
    @inbounds _to_static_voigt(A, order)
end
@inline function tovoigt(::Type{<:SVector}, A::SymmetricTensor{2, dim, T, M}; offdiagscale=one(T), order=nothing) where {dim, T, M}
    @inbounds _to_static_voigt(A, order; offdiagscale=offdiagscale)
end
@inline function tovoigt(::Type{<:SMatrix}, A::SymmetricTensor{4, dim, T}; offdiagscale=one(T), order=nothing) where {dim, T}
    @inbounds _to_static_voigt(A, order; offdiagscale=offdiagscale)
end

# default voigt order
function _to_static_voigt(A::TT, ::Nothing; offdiagscale=one(T)) where {TT<:SecondOrderTensor{dim,T}} where {dim,T}
    tuple_data, N = _to_voigt_tuple(A, offdiagscale)
    return SVector{N, T}(tuple_data)
end
function _to_static_voigt(A::TT, ::Nothing; offdiagscale=one(T)) where {TT<:FourthOrderTensor{dim,T}} where {dim,T}
    tuple_data, N = _to_voigt_tuple(A, offdiagscale)
    return SMatrix{N, N, T}(tuple_data)
end

# custom voigt order
@inline function _to_static_voigt(A::Tensor{2, dim, T, M}, order::AbstractVecOrMat) where {dim, T, M}
    v = MVector{M, T}(undef)
    @inbounds _tovoigt!(v, A, order)
    return SVector(v)
end
@inline function _to_static_voigt(A::Tensor{4, dim, T, M}, order::AbstractVecOrMat) where {dim, T, M}
    m = MMatrix{Int(√M), Int(√M), T}(undef)
    @inbounds _tovoigt!(m, A, order)
    return SMatrix(m)
end
@inline function _to_static_voigt(A::SymmetricTensor{2, dim, T, M}, order::AbstractVecOrMat; offdiagscale=one(T)) where {dim, T, M}
    v = MVector{M, T}(undef)
    @inbounds _tovoigt!(v, A, order; offdiagscale=offdiagscale)
    return SVector(v)
end
@inline function _to_static_voigt(A::SymmetricTensor{4, dim, T, M}, order::AbstractVecOrMat; offdiagscale=one(T)) where {dim, T, M}
    m = MMatrix{Int(√M), Int(√M), T}(undef)
    @inbounds _tovoigt!(m, A, order; offdiagscale=offdiagscale)
    return SMatrix(m)
end


