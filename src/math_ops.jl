# norm, det, inv, eig, trace, dev
"""
    norm(::Vec)
    norm(::SecondOrderTensor)
    norm(::FourthOrderTensor)

Computes the norm of a tensor.

# Examples
```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensor{2, 3, Float64, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> norm(A)
1.8223398556552728
```
"""
@inline LinearAlgebra.norm(v::Vec) = sqrt(dot(v, v))
@inline LinearAlgebra.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))

@generated function LinearAlgebra.norm(S::Tensor{3,dim}) where {dim}
    idx(i,j,k) = compute_index(get_base(S), i, j, k)
    ex = Expr[]
    for k in 1:dim, j in 1:dim, i in 1:dim
        push!(ex, :(get_data(S)[$(idx(i,j,k))]))
    end
    exp = reducer(ex, ex)
    return quote
      $(Expr(:meta, :inline))
      @inbounds return sqrt($exp)
    end
end

# special case for Tensor{4, 3} since it is faster than unrolling
@inline LinearAlgebra.norm(S::Tensor{4, 3}) = sqrt(mapreduce(abs2, +, S))

@generated function LinearAlgebra.norm(S::FourthOrderTensor{dim}) where {dim}
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

LinearAlgebra.normalize(t::AbstractTensor) = t/norm(t)

"""
    det(::SecondOrderTensor)

Computes the determinant of a second order tensor.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> det(A)
-0.002539324113350679
```
"""
@inline LinearAlgebra.det(t::SecondOrderTensor{1}) = @inbounds t[1,1]
@inline LinearAlgebra.det(t::SecondOrderTensor{2}) = @inbounds (t[1,1] * t[2,2] - t[1,2] * t[2,1])
@inline function LinearAlgebra.det(t::SecondOrderTensor{3})
    @inbounds (t[1,1] * (t[2,2]*t[3,3] - t[2,3]*t[3,2]) -
                  t[1,2] * (t[2,1]*t[3,3] - t[2,3]*t[3,1]) +
                  t[1,3] * (t[2,1]*t[3,2] - t[2,2]*t[3,1]))
end

"""
    inv(::SecondOrderTensor)

Computes the inverse of a second order tensor.

# Examples
```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensor{2, 3, Float64, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> inv(A)
3×3 Tensor{2, 3, Float64, 9}:
 -587.685  -279.668   1583.46
 -411.743  -199.494   1115.12
  588.35    282.819  -1587.79
```
"""
@generated function Base.inv(t::Tensor{2, dim}) where {dim}
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

@generated function Base.inv(t::SymmetricTensor{2, dim}) where {dim}
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

function Base.inv(t::Tensor{4, dim}) where {dim}
    fromvoigt(Tensor{4, dim}, inv(tovoigt(t)))
end

function Base.inv(t::SymmetricTensor{4, dim, T}) where {dim, T}
    frommandel(SymmetricTensor{4, dim}, inv(tomandel(t)))
end

function Base.inv(t::Tensor{4, dim, <:Real}) where {dim}
    fromvoigt(Tensor{4, dim}, inv(tovoigt(SMatrix, t)))
end

function Base.inv(t::SymmetricTensor{4, dim, T}) where {dim, T<:Real}
    frommandel(SymmetricTensor{4, dim}, inv(tomandel(SMatrix, t)))
end


Base.:\(S1::SecondOrderTensor, S2::AbstractTensor) = inv(S1) ⋅ S2

"""
    sqrt(S::SymmetricTensor{2})

Calculate the square root of the positive definite symmetric
second order tensor `S`, such that `√S ⋅ √S == S`.

# Examples
```jldoctest
julia> S = rand(SymmetricTensor{2,2}); S = tdot(S)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.407718  0.298993
 0.298993  0.349237

julia> sqrt(S)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.578172  0.270989
 0.270989  0.525169

julia> √S ⋅ √S ≈ S
true
```
"""
Base.sqrt(::SymmetricTensor{2})

Base.sqrt(S::SymmetricTensor{2,1}) = SymmetricTensor{2,1}((sqrt(S[1,1]),))

# https://en.m.wikipedia.org/wiki/Square_root_of_a_2_by_2_matrix
function Base.sqrt(S::SymmetricTensor{2,2})
    s = √(det(S))
    t = √(tr(S)+2s)
    return SymmetricTensor{2,2}((S[1,1]+s, S[2,1], S[2,2]+s)) / t
end

function Base.sqrt(S::SymmetricTensor{2,3,T}) where T
    E = eigen(S)
    λ = E.values
    Φ = E.vectors
    z = zero(T)
    Λ = Tensor{2,3}((√(λ[1]), z, z, z, √(λ[2]), z, z, z, √(λ[3])))
    return unsafe_symmetric(Φ⋅Λ⋅Φ')
end

"""
    tr(::SecondOrderTensor)

Computes the trace of a second order tensor.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> tr(A)
1.6144775244804341
```
"""
@generated function LinearAlgebra.tr(S::SecondOrderTensor{dim}) where {dim}
    idx(i,j) = compute_index(get_base(S), i, j)
    ex = Expr[:(get_data(S)[$(idx(i,i))]) for i in 1:dim]
    exp = reduce((ex1, ex2) -> :(+($ex1, $ex2)), ex)
    @inbounds return exp
end

Statistics.mean(S::SecondOrderTensor) = tr(S) / 3

"""
    vol(::SecondOrderTensor)

Computes the volumetric part of a second order tensor
based on the additive decomposition.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> vol(A)
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.538159  0.0       0.0
 0.0       0.538159  0.0
 0.0       0.0       0.538159

julia> vol(A) + dev(A) ≈ A
true
```
"""
vol(S::SecondOrderTensor) = mean(S) * one(S)

"""
    dev(::SecondOrderTensor)

Computes the deviatoric part of a second order tensor.

# Examples
```jldoctest
julia> A = rand(Tensor{2, 3});

julia> dev(A)
3×3 Tensor{2, 3, Float64, 9}:
 -0.065136   0.894245   0.953125
  0.549051  -0.0380011  0.795547
  0.218587   0.394255   0.103137

julia> tr(dev(A))
5.551115123125783e-17
```
"""
@inline function dev(S::SecondOrderTensor)
    Tt = get_base(typeof(S))
    trace = tr(S) / 3
    Tt(
        @inline function(i, j)
            @inbounds  v = i == j ? S[i,j] - trace : S[i,j]
            v
        end
    )
end

# https://web.archive.org/web/20150311224314/http://inside.mines.edu/fs_home/gmurray/ArbitraryAxisRotation/
"""
    rotate(x::AbstractTensor{3}, u::Vec{3}, θ::Number)

Rotate a three dimensional tensor `x` around the vector `u` a total of `θ` radians.

# Examples
```jldoctest
julia> x = Vec{3}((0.0, 0.0, 1.0));

julia> u = Vec{3}((0.0, 1.0, 0.0));

julia> rotate(x, u, π/2)
3-element Vec{3, Float64}:
 1.0
 0.0
 6.123233995736766e-17
```
"""
rotate(x::AbstractTensor{<:Any,3}, u::Vec{3}, θ::Number)

function rotate(x::Vec{3}, u::Vec{3}, θ::Number)
    ux = u ⋅ x
    u² = u ⋅ u
    s, c = sincos(θ)
    (u * ux * (1 - c) + u² * x * c + sqrt(u²) * (u × x) * s) / u²
end

"""
    rotate(x::AbstractTensor{2}, θ::Number)

Rotate a two dimensional tensor `x` `θ` radians around the out-of-plane axis.

# Examples
```jldoctest
julia> x = Vec{2}((0.0, 1.0));

julia> rotate(x, π/4)
2-element Vec{2, Float64}:
 -0.7071067811865475
  0.7071067811865476
```
"""
rotate(x::AbstractTensor{<:Any, 2}, θ::Number)

function rotate(x::Vec{2}, θ::Number)
    s, c = sincos(θ)
    return Vec{2}((c * x[1] - s * x[2], s * x[1] + c * x[2]))
end

@deprecate rotation_matrix rotation_tensor false

"""
    rotation_tensor(θ::Number)

Return the two-dimensional rotation matrix corresponding to rotation of `θ` radians around
the out-of-plane axis (i.e. around `(0, 0, 1)`).
"""
function rotation_tensor(θ::Number)
    s, c = sincos(θ)
    return Tensor{2, 2}((c, s, -s, c))
end

"""
    rotation_tensor(ψ::Number, θ::Number, ϕ::Number)

Return the three-dimensional rotation matrix corresponding to the rotation described
by the three Euler angles ``ψ``, ``θ``, ``ϕ``.

```math
R(ψ,θ,ϕ) = R_x(ψ)R_y(θ)R_z(ϕ)
```
see e.g. <http://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf> for a complete description.

Note that the [gimbal lock phenomena](https://en.wikipedia.org/wiki/Gimbal_lock) can occur when using
this rotation tensor parametrization.
"""
function rotation_tensor(ψ::Number,θ::Number,ϕ::Number)
    sψ, cψ = sincos(ψ)
    sθ, cθ = sincos(θ)
    sϕ, cϕ = sincos(ϕ)
    return Tensor{2,3}((cθ * cϕ, cθ * sϕ, -sθ,                                    #first column
                        sψ * sθ * cϕ - cψ * sϕ, sψ * sθ * sϕ + cψ * cϕ, sψ * cθ,  #second column
                        cψ * sθ * cϕ + sψ * sϕ, cψ * sθ * sϕ - sψ * cϕ, cψ * cθ)) #third column
end

"""
    rotation_tensor(u::Vec{3}, θ::Number)

Return the three-dimensional rotation matrix corresponding to rotation of `θ` radians around
the vector `u`.
"""
function rotation_tensor(u::Vec{3, T}, θ::Number) where T
    # See http://mathworld.wolfram.com/RodriguesRotationFormula.html
    u = u / norm(u)
    z = zero(T)
    ω = Tensor{2, 3}((z, u[3], -u[2], -u[3], z, u[1], u[2], -u[1], z))
    s, c = sincos(θ)
    return one(ω) + s * ω + (1 - c) * ω^2
end

# args is (u::Vec{3}, θ::Number) for 3D tensors, and (θ::number,) for 2D
function rotate(x::SymmetricTensor{2}, args...)
    R = rotation_tensor(args...)
    return unsafe_symmetric(R ⋅ x ⋅ R')
end
function rotate(x::Tensor{2}, args...)
    R = rotation_tensor(args...)
    return R ⋅ x ⋅ R'
end
function rotate(x::Tensor{3}, args...)
    R = rotation_tensor(args...)
    return otimesu(R, R) ⊡ x ⋅ R'
end
function rotate(x::Tensor{4}, args...)
    R = rotation_tensor(args...)
    return otimesu(R, R) ⊡ x ⊡ otimesu(R', R')
end
function rotate(x::SymmetricTensor{4}, args...)
    R = rotation_tensor(args...)
    return unsafe_symmetric(otimesu(R, R) ⊡ x ⊡ otimesu(R', R'))
end

isparallel(v1::Vec{3}, v2::Vec{3}) = isapprox(Tensors.cross(v1,v2), zero(v1), atol = 1e-14)
isparallel(v1::Vec{2}, v2::Vec{2}) = isapprox(v1[2]*v2[1], v1[1]*v2[2], atol = 1e-14)

"""
    orthogonal(u::Vec)

Return a orthogonal vector `v` to u (v ⋅ u = 0.0)
"""
function orthogonal(u::Vec{3}) 
    iszero(u) && error("Cannot construct a orthogonal vector to a vector with zero length")
    r = rand(u)
    while isparallel(r,u) || iszero(r) #Is this needed?
        r = rand(u)
    end
    q = r - u*(r⋅u)/(u⋅u)
    return q
end


function orthogonal(u::Vec{2}) 
    iszero(u) && error("Cannot construct a orthogonal vector to a vector with zero length")
    return Vec((u[2], -u[1]))
end

function orthogonal(u::Vec{1}) 
    error("Cannot construct a orthogonal vector for a one dimensional vector")
    #return Vec((0.0,))
end