# dcontract, otimes, dot, tdot, cross
"""
    dcontract(::AbstractTensor{order_a}, ::AbstractTensor{order_b}) where {order_a, order_b}

Compute the double contraction between two tensors for `order_a ≥ 2` and `order_b ≥ 2`.
The symbol `⊡`, written `\\boxdot`, is overloaded for double contraction.
The reason `:` is not used is because it does not have the same precedence as multiplication.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
0.7654348606012742

julia> A ⊡ B
0.7654348606012742
```
"""
function dcontract end 

const ⊡ = dcontract

# 2-2
@generated function dcontract(A::SecondOrderTensor, B::SecondOrderTensor)
    expr = Tensors.get_expression((), :(A[i, j] * B[i, j]), (;A, B); use_muladd = false)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-3
@generated function dcontract(A::SecondOrderTensor, B::AbstractTensor{3}) 
    expr = Tensors.get_expression((:k,), :(A[i, j] * B[i, j, k]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-4
@generated function dcontract(A::SecondOrderTensor, B::FourthOrderTensor) 
    expr = Tensors.get_expression((:k, :l), :(A[i, j] * B[i, j, k, l]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 3-2
@generated function dcontract(A::AbstractTensor{3}, B::SecondOrderTensor) 
    expr = Tensors.get_expression((:i,), :(A[i, j, k] * B[j, k]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 3-3
@generated function dcontract(A::AbstractTensor{3}, B::AbstractTensor{3}) 
    expr = Tensors.get_expression((:i, :j), :(A[i, k, l] * B[k, l, j]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 3-4
@generated function dcontract(A::AbstractTensor{3}, B::FourthOrderTensor) 
    expr = Tensors.get_expression((:i, :j, :k), :(A[i, l, m] * B[l, m, j, k]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 4-2
@generated function dcontract(A::FourthOrderTensor, B::SecondOrderTensor) 
    expr = Tensors.get_expression((:i, :j), :(A[i, j, k, l] * B[k, l]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 4-3
@generated function dcontract(A::FourthOrderTensor, B::AbstractTensor{3}) 
    expr = Tensors.get_expression((:i, :j, :k), :(A[i, j, l, m] * B[l, m, k]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 4-4
@generated function dcontract(A::FourthOrderTensor, B::FourthOrderTensor) 
    expr = Tensors.get_expression((:i, :j, :k, :l), :(A[i, j, m, n] * B[m, n, k, l]), (;A, B); use_muladd = true)
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end

"""
    otimes(::Vec, ::Vec)
    otimes(::SecondOrderTensor, ::SecondOrderTensor)

Compute the open product between two tensors.
The symbol `⊗`, written `\\otimes`, is overloaded for tensor products.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> A ⊗ B
2×2×2×2 SymmetricTensor{4, 2, Float64, 9}:
[:, :, 1, 1] =
 0.291503  0.490986
 0.490986  0.19547

[:, :, 2, 1] =
 0.115106  0.193876
 0.193876  0.0771855

[:, :, 1, 2] =
 0.115106  0.193876
 0.193876  0.0771855

[:, :, 2, 2] =
 0.128518  0.216466
 0.216466  0.086179
```
"""
function otimes end 

@inline otimes(S1::Number, S2::Number) = S1*S2
@inline otimes(S1::AbstractTensor, S2::Number) = S1*S2
@inline otimes(S1::Number, S2::AbstractTensor) = S1*S2

const ⊗ = otimes

# 1-1
@generated function otimes(A::Vec, B::Vec)
    expr = get_expression((:i, :j), :(A[i] * B[j]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 1-2
@generated function otimes(A::Vec, B::SecondOrderTensor)
    expr = get_expression((:i, :j, :k), :(A[i] * B[j, k]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-1
@generated function otimes(A::SecondOrderTensor, B::Vec)
    expr = get_expression((:i, :j, :k), :(A[i, j] * B[k]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-2
@generated function otimes(A::SecondOrderTensor, B::SecondOrderTensor)
    expr = get_expression((:i, :j, :k, :l), :(A[i, j] * B[k, l]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# Not yet implemented: 1-3 and 3-1

"""
    otimes(::Vec)

Compute the open product of a vector with itself.
Return a `SymmetricTensor`.

# Examples
```jldoctest
julia> A = rand(Vec{2})
2-element Vec{2, Float64}:
 0.32597672886359486
 0.5490511363155669

julia> otimes(A)
2×2 SymmetricTensor{2, 2, Float64, 3}:
 0.106261  0.178978
 0.178978  0.301457
```
"""
@inline function otimes(S::Vec{dim}) where {dim}
    return SymmetricTensor{2, dim}(@inline function(i,j) @inbounds S[i] * S[j]; end)
end

"""
    otimesu(::SecondOrderTensor, ::SecondOrderTensor)

Compute the "upper" open product between two tensors.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> otimesu(A, B)
2×2×2×2 Tensor{4, 2, Float64, 16}:
[:, :, 1, 1] =
 0.291503  0.115106
 0.490986  0.193876

[:, :, 2, 1] =
 0.490986  0.193876
 0.19547   0.0771855

[:, :, 1, 2] =
 0.115106  0.128518
 0.193876  0.216466

[:, :, 2, 2] =
 0.193876   0.216466
 0.0771855  0.086179
```
"""
@generated function otimesu(S1::SecondOrderTensor, S2::SecondOrderTensor)
    expr = get_expression((:i, :j, :k, :l), :(S1[i, k] * S2[j, l]), (;S1, S2))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end

"""
    otimesl(::SecondOrderTensor, ::SecondOrderTensor)

Compute the "lower" open product between two tensors.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> otimesl(A, B)
2×2×2×2 Tensor{4, 2, Float64, 16}:
[:, :, 1, 1] =
 0.291503  0.115106
 0.490986  0.193876

[:, :, 2, 1] =
 0.115106  0.128518
 0.193876  0.216466

[:, :, 1, 2] =
 0.490986  0.193876
 0.19547   0.0771855

[:, :, 2, 2] =
 0.193876   0.216466
 0.0771855  0.086179
```
"""
@generated function otimesl(S1::SecondOrderTensor, S2::SecondOrderTensor)
    expr = get_expression((:i, :j, :k, :l), :(S1[i, l] * S2[j, k]), (;S1, S2))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end


"""
    dot(::Vec, ::Vec)
    dot(::Vec, ::SecondOrderTensor)
    dot(::SecondOrderTensor, ::Vec)
    dot(::SecondOrderTensor, ::SecondOrderTensor)

Computes the dot product (single contraction) between two tensors.
The symbol `⋅`, written `\\cdot`, is overloaded for single contraction.

# Examples
```jldoctest
julia> A = rand(Tensor{2, 2})
2×2 Tensor{2, 2, Float64, 4}:
 0.325977  0.218587
 0.549051  0.894245

julia> B = rand(Tensor{1, 2})
2-element Vec{2, Float64}:
 0.35311164439921205
 0.39425536741585077

julia> dot(A, B)
2-element Vec{2, Float64}:
 0.2012851406726999
 0.5464374094589712

julia> A ⋅ B
2-element Vec{2, Float64}:
 0.2012851406726999
 0.5464374094589712
```
"""
LinearAlgebra.dot(::AbstractTensor, ::AbstractTensor)

# 1-1
@generated function LinearAlgebra.dot(A::Vec, B::Vec) 
    expr = get_expression((), :(A[i] * B[i]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 1-2
@generated function LinearAlgebra.dot(A::Vec, B::SecondOrderTensor) 
    expr = get_expression((:i,), :(A[j] * B[j, i]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 1-3
@generated function LinearAlgebra.dot(A::Vec, B::AbstractTensor{3}) 
    expr = get_expression((:i, :j), :(A[k] * B[k, i, j]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-1
@generated function LinearAlgebra.dot(A::SecondOrderTensor, B::Vec) 
    expr = get_expression((:i,), :(A[i, j] * B[j]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-2
@generated function LinearAlgebra.dot(A::SecondOrderTensor, B::SecondOrderTensor)
    expr = get_expression((:i, :j), :(A[i, k] * B[k, j]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-3
@generated function LinearAlgebra.dot(A::SecondOrderTensor, B::AbstractTensor{3}) 
    expr = get_expression((:i, :j, :k), :(A[i, l] * B[l, j, k]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 2-4
@generated function LinearAlgebra.dot(A::SecondOrderTensor, B::FourthOrderTensor) 
    expr = get_expression((:i, :j, :k, :l), :(A[i, m] * B[m, j, k, l]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 3-1
@generated function LinearAlgebra.dot(A::AbstractTensor{3}, B::Vec) 
    expr = get_expression((:i, :j), :(A[i, j, k] * B[k]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 3-2
@generated function LinearAlgebra.dot(A::AbstractTensor{3}, B::SecondOrderTensor) 
    expr = get_expression((:i, :j, :k), :(A[i, j, l] * B[l, k]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end
# 4-2
@generated function LinearAlgebra.dot(A::FourthOrderTensor, B::SecondOrderTensor) 
    expr = get_expression((:i, :j, :k, :l), :(A[i, j, k, m] * B[m, l]), (;A, B))
    return quote
        $(Expr(:meta, :inline))
        @inbounds return $expr
    end
end

# Not included yet: 1-4, 4-1, 3-3

# Ensure that we don't fall back to the default implementation for AbstractArray, 
# which has a different meaning except in the case for `Vec`. 
function LinearAlgebra.dot(ta::AbstractTensor, tb::AbstractTensor)
    TA = get_base(typeof(ta))
    TB = get_base(typeof(tb))
    throw(ArgumentError("single contraction not implemented between $TA and $TB"))
end

"""
    dot(::SymmetricTensor{2})

Compute the dot product of a symmetric second order tensor with itself.
Return a `SymmetricTensor`.

See also [`tdot`](@ref) and [`dott`](@ref).

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.325977  0.549051  0.218587
 0.549051  0.894245  0.353112
 0.218587  0.353112  0.394255

julia> dot(A)
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.455498  0.74715  0.351309
 0.74715   1.22582  0.575
 0.351309  0.575    0.327905
```
"""
@inline LinearAlgebra.dot(S::SymmetricTensor{2}) = tdot(S)

"""
    tdot(A::SecondOrderTensor)

Compute the transpose-dot product of `A` with itself, i.e. `dot(A', A)`.
Return a `SymmetricTensor`.

# Examples
```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensor{2, 3, Float64, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> tdot(A)
3×3 SymmetricTensor{2, 3, Float64, 6}:
 0.455498  0.571559  0.855529
 0.571559  1.0798    1.3281
 0.855529  1.3281    1.78562
```
"""
@inline tdot(S::SecondOrderTensor) = unsafe_symmetric(S' ⋅ S)

"""
    dott(A::SecondOrderTensor)

Compute the dot-transpose product of `A` with itself, i.e. `dot(A, A')`.
Return a `SymmetricTensor`.

# Examples
```jldoctest
julia> A = rand(Tensor{2,3})
3×3 Tensor{2, 3, Float64, 9}:
 0.325977  0.894245  0.953125
 0.549051  0.353112  0.795547
 0.218587  0.394255  0.49425

julia> dott(A)
3×3 SymmetricTensor{2, 3, Float64, 6}:
 1.81438   1.253    0.894897
 1.253     1.05904  0.65243
 0.894897  0.65243  0.4475
```
"""
@inline dott(S::SecondOrderTensor) = unsafe_symmetric(S ⋅ S')

"""
    cross(::Vec, ::Vec)

Computes the cross product between two `Vec` vectors, returns a `Vec{3}`. For dimensions 1 and 2 the `Vec`'s
are expanded to 3D first. The infix operator `×` (written `\\times`) can also be used.

# Examples
```jldoctest
julia> a = rand(Vec{3})
3-element Vec{3, Float64}:
 0.32597672886359486
 0.5490511363155669
 0.21858665481883066

julia> b = rand(Vec{3})
3-element Vec{3, Float64}:
 0.8942454282009883
 0.35311164439921205
 0.39425536741585077

julia> a × b
3-element Vec{3, Float64}:
  0.13928086435138393
  0.0669520417303531
 -0.37588028973385323
```
"""
@inline LinearAlgebra.cross(u::Vec{3}, v::Vec{3}) = @inbounds Vec{3}((u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]))
@inline LinearAlgebra.cross(u::Vec{2,T1}, v::Vec{2,T2}) where {T1,T2} = @inbounds Vec{3}((zero(T1)*zero(T2), zero(T1)*zero(T2), u[1]*v[2] - u[2]*v[1]))
@inline LinearAlgebra.cross( ::Vec{1,T1}, ::Vec{1,T2}) where {T1,T2} = @inbounds zero(Vec{3,promote_type(T1,T2)})
