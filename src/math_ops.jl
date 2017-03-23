# norm, det, inv, eig, trace, dev
"""
```julia
norm(::Vec)
norm(::SecondOrderTensor)
norm(::FourthOrderTensor)
```
Computes the norm of a tensor.

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> norm(A)
1.7377443667834924
```
"""
@inline Base.norm(v::Vec) = sqrt(dot(v, v))
@inline Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))

# special case for Tensor{4, 3} since it is faster than unrolling
@inline Base.norm(S::Tensor{4, 3}) = sqrt(mapreduce(abs2, +, S))

@generated function Base.norm{dim}(S::FourthOrderTensor{dim})
    idx(i,j,k,l) = compute_index(get_base(S), i, j, k, l)
    ex = Expr[]
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        push!(ex, :(get_data(S)[$(idx(i,j,k,l))]))
    end
    exp = reducer(ex, ex)
    return quote
      $(Expr(:meta, :inline))
      @inbounds return sqrt($exp)
    end
end

"""
```julia
det(::SecondOrderTensor)
```
Computes the determinant of a second order tensor.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 Tensors.SymmetricTensor{2,3,Float64,6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> det(A)
-0.1005427219925894
```
"""
@inline Base.det(t::SecondOrderTensor{1}) = @inboundsret t[1,1]
@inline Base.det(t::SecondOrderTensor{2}) = @inboundsret (t[1,1] * t[2,2] - t[1,2] * t[2,1])
@inline function Base.det(t::SecondOrderTensor{3})
    @inboundsret (t[1,1] * (t[2,2]*t[3,3] - t[2,3]*t[3,2]) -
                  t[1,2] * (t[2,1]*t[3,3] - t[2,3]*t[3,1]) +
                  t[1,3] * (t[2,1]*t[3,2] - t[2,2]*t[3,1]))
end

"""
```julia
inv(::SecondOrderTensor)
```
Computes the inverse of a second order tensor.

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> inv(A)
3×3 Tensors.Tensor{2,3,Float64,9}:
  19.7146   -19.2802    7.30384
   6.73809  -10.7687    7.55198
 -68.541     81.4917  -38.8361
```
"""
@generated function Base.inv{dim}(t::Tensor{2, dim})
    Tt = get_base(t)
    idx(i,j) = compute_index(Tt, i, j)
    if dim == 1
        ex = :($Tt((dinv, )))
    elseif dim == 2
        ex = quote
            v = get_data(t)
            $Tt((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                -v[$(idx(1,2))] * dinv,  v[$(idx(1,1))] * dinv))
        end
    else # dim == 3
        ex = quote
            v = get_data(t)
            $Tt(((v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                 (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                -(v[$(idx(1,2))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,2))]) * dinv,
                 (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                 (v[$(idx(1,2))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,2))]) * dinv,
                -(v[$(idx(1,1))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,1))]) * dinv,
                 (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        dinv = 1 / det(t)
        @inbounds return $ex
    end
end

@generated function Base.inv{dim}(t::SymmetricTensor{2, dim})
    Tt = get_base(t)
    idx(i,j) = compute_index(Tt, i, j)
    if dim == 1
        ex = :($Tt((dinv, )))
    elseif dim == 2
        ex = quote
            v = get_data(t)
            $Tt((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                 v[$(idx(1,1))] * dinv))
        end
    else # dim == 3
        ex = quote
            v = get_data(t)
            $Tt(((v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                 (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                 (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                 (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv))
        end
    end
    return quote
        $(Expr(:meta, :inline))
        dinv = 1 / det(t)
        @inbounds return $ex
    end
end

function Base.inv{dim}(t::Tensor{4, dim})
    fromvoigt(Tensor{4, dim}, inv(tovoigt(t)))
end

function Base.inv{dim, T}(t::SymmetricTensor{4, dim, T})
    # use mandel form to reduce order of symmetric tensor
    s = T(√2)
    fromvoigt(SymmetricTensor{4, dim}, inv(tovoigt(t, offdiagscale=s)), offdiagscale=s)
end

"""
```julia
eig(::SymmetricTensor{2})
```
Compute the eigenvalues and eigenvectors of a symmetric second order tensor.
`eig` is a wrapper around [`eigfact`](@ref) which extracts eigenvalues and
eigenvectors to a tuple.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2})
2×2 Tensors.SymmetricTensor{2,2,Float64,3}:
 0.590845  0.766797
 0.766797  0.566237

julia> Λ, Φ = eig(A);

julia> Λ
2-element Tensors.Tensor{1,2,Float64,2}:
 -0.188355
  1.34544

julia> Φ
2×2 Tensors.Tensor{2,2,Float64,4}:
 -0.701412  0.712756
  0.712756  0.701412

julia> Φ ⋅ diagm(Tensor{2, 2}, Λ) ⋅ inv(Φ) # Same as A
2×2 Tensors.Tensor{2,2,Float64,4}:
 0.590845  0.766797
 0.766797  0.566237
```
"""
@inline Base.eig(S::SymmetricTensor) = (E = eigfact(S); (E.λ, E.Φ))

"""
```julia
eigvals(::SymmetricTensor{2})
```
Compute the eigenvalues of a symmetric second order tensor.
"""
@inline Base.eigvals(S::SymmetricTensor) = (E = eigfact(S); E.λ)

"""
```julia
eigvecs(::SymmetricTensor{2})
```
Compute the eigenvectors of a symmetric second order tensor.
"""
@inline Base.eigvecs(S::SymmetricTensor) = (E = eigfact(S); E.Φ)

immutable Eigen{T, dim, M}
    λ::Vec{dim, T}
    Φ::Tensor{2, dim, T, M}
end

"""
```julia
eigfact(::SymmetricTensor{2})
```
Compute the eigenvalues and eigenvectors of a symmetric second order tensor
and return an `Eigen` object. The eigenvalues are stored in a `Vec`,
sorted in ascending order. The corresponding eigenvectors are stored
as the columns of a `Tensor`.

See [`eigvals`](@ref) and [`eigvecs`](@ref).

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2})
2×2 Tensors.SymmetricTensor{2,2,Float64,3}:
 0.590845  0.766797
 0.766797  0.566237

julia> E = eigfact(A)
Tensors.Eigen{Float64,2,4}([-0.188355, 1.34544], [-0.701412 0.712756; 0.712756 0.701412])

julia> eigvals(E)
2-element Tensors.Tensor{1,2,Float64,2}:
 -0.188355
  1.34544

julia> eigvecs(E)
2×2 Tensors.Tensor{2,2,Float64,4}:
 -0.701412  0.712756
  0.712756  0.701412
```
"""
Base.eigfact

"""
```julia
eigvals(::Eigen)
```
Extract eigenvalues from an `Eigen` object, returned by [`eigfact`](@ref).
"""
@inline Base.eigvals(E::Eigen) = E.λ
"""
```julia
eigvecs(::Eigen)
```
Extract eigenvectors from an `Eigen` object, returned by [`eigfact`](@ref).
"""
@inline Base.eigvecs(E::Eigen) = E.Φ

"""
```julia
trace(::SecondOrderTensor)
```
Computes the trace of a second order tensor.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 Tensors.SymmetricTensor{2,3,Float64,6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> trace(A)
1.9050765715072775
```
"""
@generated function Base.trace{dim}(S::SecondOrderTensor{dim})
    idx(i,j) = compute_index(get_base(S), i, j)
    ex = Expr[:(get_data(S)[$(idx(i,i))]) for i in 1:dim]
    exp = reduce((ex1, ex2) -> :(+($ex1, $ex2)), ex)
    @inbounds return exp
end

Base.mean(S::SecondOrderTensor) = trace(S) / 3

"""
```julia
vol(::SecondOrderTensor)
```
Computes the volumetric part of a second order tensor based on the additive decomposition.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 Tensors.SymmetricTensor{2,3,Float64,6}:
 0.0339856  0.347344  0.416272
 0.347344   0.545664  0.885307
 0.416272   0.885307  0.683448

julia> vol(A)
3×3 Tensors.SymmetricTensor{2,3,Float64,6}:
 0.421032  0.0       0.0
 0.0       0.421032  0.0
 0.0       0.0       0.421032

julia> vol(A) + dev(A) ≈ A
true
```
"""
vol(S::SecondOrderTensor) = mean(S) * one(S)

"""
```julia
dev(::SecondOrderTensor)
```
Computes the deviatoric part of a second order tensor.

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3});

julia> dev(A)
3×3 Tensors.Tensor{2,3,Float64,9}:
 0.0469421  0.460085   0.200586
 0.766797   0.250123   0.298614
 0.566237   0.854147  -0.297065

julia> trace(dev(A))
0.0
```
"""
@inline function dev(S::SecondOrderTensor)
    Tt = get_base(typeof(S))
    tr = trace(S) / 3
    Tt(
        @inline function(i, j)
            @inbounds  v = i == j ? S[i,j] - tr : S[i,j]
            v
        end
    )
end

# http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
"""
Rotate a three dimensional vector `x` around another vector `u` a total of `θ` radians.

```julia
rotate(x::Vec{3}, u::Vec{3}, θ::Number)
```

**Example:**

```jldoctest
julia> x = Vec{3}((0.0, 0.0, 1.0))
3-element Tensors.Tensor{1,3,Float64,3}:
 0.0
 0.0
 1.0

julia> u = Vec{3}((0.0, 1.0, 0.0))
3-element Tensors.Tensor{1,3,Float64,3}:
 0.0
 1.0
 0.0

julia> rotate(x, u, π/2)
3-element Tensors.Tensor{1,3,Float64,3}:
 1.0
 0.0
 6.12323e-17
```
"""
function rotate(x::Vec{3}, u::Vec{3}, θ::Number)
    ux = u ⋅ x
    u² = u ⋅ u
    c = cos(θ)
    s = sin(θ)
    (u * ux * (1 - c) + u² * x * c + sqrt(u²) * (u × x) * s) / u²
end

