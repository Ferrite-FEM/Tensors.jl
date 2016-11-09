"""
Computes the double contraction between two tensors.
The symbol `⊡`, written `\\boxdot`, is overloaded for double contraction.
The reason `:` is not used is because it does not have the same precedence as multiplication.

```julia
dcontract(::SecondOrderTensor, ::SecondOrderTensor)
dcontract(::SecondOrderTensor, ::FourthOrderTensor)
dcontract(::FourthOrderTensor, ::SecondOrderTensor)
dcontract(::FourthOrderTensor, ::FourthOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> dcontract(A,B)
1.9732018397544984

julia> A ⊡ B
1.9732018397544984
```
"""
@inline dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim}) = tovector(S1) ⋅ tovector(S2)

@inline function dcontract{dim, T1, T2, M}(S1::Tensor{4, dim, T1, M}, S2::Tensor{4, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    Tensor{4, dim, Tv, M}(tomatrix(S1) * tomatrix(S2))
end

@inline function dcontract{dim, T1, T2, M}(S1::Tensor{4, dim, T1}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1) * zero(T2))
    Tensor{2, dim, Tv, M}(tomatrix(S1) * tovector(S2))
end

@inline function dcontract{dim,T1,T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{4, dim, T2})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{2, dim, Tv, M}(tomatrix(S2)' * tovector(S1))
end

const ⊡ = dcontract

# Promotion
@inline dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{2, dim}) = dcontract(S1, convert(Tensor, S2))
@inline dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{4, dim}) = dcontract(S1, convert(Tensor, S2))

@inline dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{2, dim}) = dcontract(convert(Tensor, S1), S2)
@inline dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{4, dim}) = dcontract(convert(Tensor, S1), S2)

@inline dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{4, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{4, dim}) = dcontract(promote(S1, S2)...)


"""
Computes the norm of a tensor

```julia
norm(::Vec)
norm(::SecondOrderTensor)
norm(::FourthOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3})
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> norm(A)
1.7377443667834922
```
"""
@inline Base.norm(v::Vec) = sqrt(dot(v, v))
@inline Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))
@inline Base.norm{dim, T}(S::Tensor{4, dim, T}) = sqrt(sumabs2(tovector(S)))

"""
Computes the open product between two tensors.
The symbol `⊗`, written `\\otimes`, is overloaded for tensor products.

```julia
otimes(::Vec, ::Vec)
otimes(::SecondOrderTensor, ::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> B = rand(SymmetricTensor{2, 2});

julia> A ⊗ B
2×2×2×2 ContMechTensors.SymmetricTensor{4,2,Float64,9}:
[:, :, 1, 1] =
 0.271839  0.352792
 0.352792  0.260518

[:, :, 2, 1] =
 0.469146  0.608857
 0.608857  0.449607

[:, :, 1, 2] =
 0.469146  0.608857
 0.608857  0.449607

[:, :, 2, 2] =
 0.504668  0.654957
 0.654957  0.48365
```
"""
@inline function otimes{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    N = n_components(Tensor{4, dim})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{4, dim, Tv, N}(tovector(S1) * tovector(S2)')
end

@inline function otimes{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2})
    N = n_components(Tensor{2, dim})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{2, dim, Tv, N}(tovector(v1) * tovector(v2)')
end

const ⊗ = otimes

# Promotion
@inline otimes{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = otimes(promote(S1, S2)...)
@inline otimes{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = otimes(promote(S1, S2)...)

"""
Computes the dot product (single contraction) between two tensors.
The symbol `⋅`, written `\\cdot`, is overloaded for single contraction.

```julia
dot(::Vec, ::Vec)
dot(::Vec, ::SecondOrderTensor)
dot(::SecondOrderTensor, ::Vec)
dot(::SecondOrderTensor, ::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2, 2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> B = rand(Tensor{1, 2})
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.794026
 0.854147

julia> dot(A, B)
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.952796
 1.00184

julia> A ⋅ B
2-element ContMechTensors.Tensor{1,2,Float64,2}:
 0.952796
 1.00184
```
"""
@inline Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = tovector(v1) ⋅ tovector(v2)

@inline function Base.dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(tomatrix(S1) * tovector(v2))
end

@inline function Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::Tensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(tomatrix(S2)' * tovector(v1))
end

@inline function Base.dot{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1) * zero(T2))
    return Tensor{2, dim, Tv, M}(tomatrix(S1) * tomatrix(S2))
end

@inline function Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    S1_t = convert(Tensor{2, dim}, S1)
    S2_t = convert(Tensor{2, dim}, S2)
    return Tensor{2, dim}(tomatrix(S1_t) * tomatrix(S2_t))
end

# Promotion
Base.dot{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dot(promote(S1, S2)...)
Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dot(promote(S1, S2)...)

"""
Computes the transpose-dot product (single contraction) between two tensors.

```julia
tdot(::Vec, ::Vec)
tdot(::Vec, ::SecondOrderTensor)
tdot(::SecondOrderTensor, ::Vec)
tdot(::SecondOrderTensor, ::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> B = rand(Tensor{2,2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.794026  0.200586
 0.854147  0.298614

julia> tdot(A,B)
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 1.1241    0.347492
 0.842587  0.250967

julia> A'⋅B
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 1.1241    0.347492
 0.842587  0.250967
```
"""
@inline tdot{dim, T1, T2}(v1::Vec{dim, T1}, S2::SecondOrderTensor{dim, T2}) = dot(v1, S2)
@inline tdot{dim, T1, T2}(S1::SecondOrderTensor{dim, T1}, v2::Vec{dim, T2}) = dot(v2, S1)
@inline tdot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = dot(v1, v2)

@inline function tdot{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1) * zero(T2))
    return Tensor{2, dim, Tv, M}(tomatrix(S1)' * tomatrix(S2))
end

@inline tdot{dim, T1, T2, M}(S1::SymmetricTensor{2, dim, T1, M}, S2::SymmetricTensor{2, dim, T2, M}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::SymmetricTensor{2, dim, T1, M1}, S2::Tensor{2, dim, T2, M2}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::Tensor{2, dim, T1, M1}, S2::SymmetricTensor{2, dim, T2, M2}) = tdot(promote(S1,S2)...)

"""
Computes the transpose-dot of a second order tensor with itself.
Returns a `SymmetricTensor`

```julia
tdot(::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3})
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> tdot(A)
3×3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 1.2577   1.36435   0.48726
 1.36435  1.57172   0.540229
 0.48726  0.540229  0.190334
```
"""
@inline function tdot{dim}(S1::Tensor{2, dim})
    return SymmetricTensor{2, dim}(transpdot(get_data(S1)))
end
@inline tdot{dim}(S1::SymmetricTensor{2,dim}) = tdot(convert(Tensor{2,dim}, S1))


"""
Computes the trace of a second order tensor.
The synonym `vol` can also be used.

```julia
trace(::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> trace(A)
1.9050765715072775
```
"""
@generated function Base.trace{dim, T}(S::SecondOrderTensor{dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(S), i, j)
    exp = Expr(:call)
    push!(exp.args, :+)
    for i in 1:dim
        push!(exp.args, :(get_data(S)[$(idx(i,i))]))
    end
    return exp
end
vol(S::SecondOrderTensor) = trace(S)


########
# Mean #
########
Base.mean(S::SecondOrderTensor) = trace(S) / 3


"""
Computes the determinant of a second order tensor.

```julia
det(::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> det(A)
-0.1005427219925894
```
"""
@gen_code function Base.det{dim, T}(t::SecondOrderTensor{dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :($(Expr(:meta, :inline)))
    @code :(v = get_data(t))
    if dim == 1
        @code :(@inbounds d = v[$(idx(1,1))])
    elseif dim == 2
        @code :(@inbounds d = v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))] * v[$(idx(2,1))])
    else
        @code :(@inbounds d = ( v[$(idx(1,1))]*(v[$(idx(2,2))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,2))]) -
                                v[$(idx(1,2))]*(v[$(idx(2,1))]*v[$(idx(3,3))]-v[$(idx(2,3))]*v[$(idx(3,1))]) +
                                v[$(idx(1,3))]*(v[$(idx(2,1))]*v[$(idx(3,2))]-v[$(idx(2,2))]*v[$(idx(3,1))])))
    end
    @code :(return d)
end


"""
Computes the inverse of a second order tensor.

```julia
inv(::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3})
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.590845  0.460085  0.200586
 0.766797  0.794026  0.298614
 0.566237  0.854147  0.246837

julia> inv(A)
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
  19.7146   -19.2802    7.30384
   6.73809  -10.7687    7.55198
 -68.541     81.4917  -38.8361
```
"""
@gen_code function Base.inv{dim, T}(t::Tensor{2, dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :($(Expr(:meta, :inline)))
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return  typeof(t)((dinv,)))
    elseif dim == 2
        @code :(return typeof(t)((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                 -v[$(idx(1,2))] * dinv, v[$(idx(1,1))] * dinv)))
    else
        @code :(return typeof(t)((  (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                                   -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                                    (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                                   -(v[$(idx(1,2))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,2))]) * dinv,
                                    (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,2))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,2))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(2,3))] - v[$(idx(1,3))]*v[$(idx(2,1))]) * dinv,
                                    (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv)))
    end
end


#######
# Dev #
#######

"""
Computes the deviatoric part of a second order tensor.
"""
@generated function dev{dim, T, M}(S::Tensor{2, dim, T, M})
    f = (i,j) -> i == j ? :((get_data(S)[$(compute_index(Tensor{2, dim}, i, j))] - tr/3)) :
                           :(get_data(S)[$(compute_index(Tensor{2, dim}, i, j))])
    exp = tensor_create(Tensor{2, dim, T}, f)
    Tv = typeof(zero(T) * 1 / 3)
    return quote
        $(Expr(:meta, :inline))
        tr = trace(S)
        Tensor{2, dim, $Tv, M}($exp)
    end
end

"""
Permutes the dimensions according to `idx` of a fourth order tensor.

```julia
permutedims(::FourthOrderTensor, idx::NTuple{4,Int})
```
"""
function Base.permutedims{dim}(S::FourthOrderTensor{dim}, idx::NTuple{4,Int})
    sort([idx...]) == [1,2,3,4] || throw(ArgumentError("Missing index."))
    neworder = sortperm([idx...])
    f = (i,j,k,l) -> S[[i,j,k,l][neworder]...]
    return Tensor{4,dim}(f)
end


"""
Computes the transpose of a tensor.
For a fourth order tensor, the transpose is the minor transpose

```julia
transpose(::Vec)
transpose(::SecondOrderTensor)
transpose(::FourthOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> A'
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.766797
 0.566237  0.460085
```
"""
@inline Base.transpose(S::Vec) = S

@inline function Base.transpose(S::Tensor{2})
    typeof(S)(tomatrix(S).')
end
Base.transpose(S::SymmetricTensor{2}) = S

"""
Computes the minor transpose of a fourth order tensor.

```julia
minortranspose(::FourthOrderTensor)
```
"""
@generated function minortranspose{dim, T, M}(t::Tensor{4, dim, T, M})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        push!(exps, :(get_data(t)[$(compute_index(Tensor{4, dim}, j, i, l, k))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            Tensor{4, dim, T, M}($exp)
        end
end
##############################
minortranspose(S::SymmetricTensor{4}) = S
Base.transpose(S::FourthOrderTensor) = minortranspose(S)

"""
Computes the major transpose of a fourth order tensor.

```julia
majortranspose(::FourthOrderTensor)
```
"""
@generated function majortranspose{dim, T}(t::FourthOrderTensor{dim, T})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        push!(exps, :(get_data(t)[$(compute_index(get_base(t), k, l, i, j))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            Tensor{4, dim, T, $N}($exp)
        end
end

Base.ctranspose(S::AllTensors) = transpose(S)


"""
Computes the symmetric part of a second or fourth order tensor.
For a fourth order tensor, the symmetric part is the same as the minor symmetric part.
Returns a `SymmetricTensor`.

```julia
symmetric(::SecondOrderTensor)
symmetric(::FourthOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,2})
2×2 ContMechTensors.Tensor{2,2,Float64,4}:
 0.590845  0.566237
 0.766797  0.460085

julia> symmetric(A)
2×2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 0.590845  0.666517
 0.666517  0.460085
```
"""
@inline symmetric(S1::SymmetricTensors) = S1

@generated function symmetric{dim, T}(t::Tensor{2, dim, T})
    N = n_components(SymmetricTensor{2, dim})
    rows = Int(div(sqrt(1 + 8*N), 2))
    exps = Expr[]
    for row in 1:rows, col in row:rows
        if row == col
            push!(exps, :(get_data(t)[$(compute_index(Tensor{2, dim}, row, col))]))
        else
            I = compute_index(Tensor{2, dim}, row, col)
            J = compute_index(Tensor{2, dim}, col, row)
            push!(exps, :((get_data(t)[$I] + get_data(t)[$J])/2))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            SymmetricTensor{2, dim, T, $N}($exp)
        end
end

"""
Computes the minor symmetric part of a fourth order tensor, returns a `SymmetricTensor{4}`

```julia
minorsymmetric(::FourthOrderTensor)
```
"""
@generated function minorsymmetric{dim, T}(t::Tensor{4, dim, T})
    N = n_components(Tensor{4, dim})
    M = n_components(SymmetricTensor{4,dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for k in 1:rows, l in k:rows, i in 1:rows, j in i:rows
        if i == j && k == l
            push!(exps, :(get_data(t)[$(compute_index(Tensor{4, dim}, i, j, k, l))]))
        else
            I = compute_index(Tensor{4, dim}, i, j, k, l)
            J = compute_index(Tensor{4, dim}, j, i, k, l)
            K = compute_index(Tensor{4, dim}, i, j, k, l)
            L = compute_index(Tensor{4, dim}, i, j, l, k)
            push!(exps, :((get_data(t)[$I] + get_data(t)[$J] + get_data(t)[$K] + get_data(t)[$L]) / 4))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            SymmetricTensor{4, dim, T, $M}($exp)
        end
end

@inline minorsymmetric(t::SymmetricTensors) = t

@inline symmetric(t::Tensor{4}) = minorsymmetric(t)

"""
Computes the major symmetric part of a fourth order tensor, returns a `Tensor{4}`

```julia
majorsymmetric(::FourthOrderTensor)
```
"""
@generated function majorsymmetric{dim, T}(t::FourthOrderTensor{dim, T})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        if i == j == k == l || i == k && j == l
            push!(exps, :(get_data(t)[$(compute_index(get_base(t), i, j, k, l))]))
        else
            I = compute_index(get_base(t), i, j, k, l)
            J = compute_index(get_base(t), k, l, i, j)
            push!(exps, :((get_data(t)[$I] + get_data(t)[$J]) / 2))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            Tensor{4, dim, T, $N}($exp)
        end
end


"""
Computes the skew-symmetric (anti-symmetric) part of a second order tensor, returns a `Tensor{2}`

```julia
skew(::SecondOrderTensor)
```
"""
@inline skew(S1::Tensor{2}) = (S1 - S1.') / 2
@inline skew{dim,T}(S1::SymmetricTensor{2,dim,T}) = zero(Tensor{2,dim,T})


"""
Computes the cross product between two `Vec` vectors, returns a `Vec{3}`. For dimensions 1 and 2 the `Vec`'s
are expanded to 3D first. The infix operator × (written `\\times`) can also be used

```julia
cross(::Vec, ::Vec)
```

**Example:**

```jldoctest
julia> a = rand(Vec{3})
3-element ContMechTensors.Tensor{1,3,Float64,3}:
 0.590845
 0.766797
 0.566237

julia> b = rand(Vec{3})
3-element ContMechTensors.Tensor{1,3,Float64,3}:
 0.460085
 0.794026
 0.854147

julia> a × b
3-element ContMechTensors.Tensor{1,3,Float64,3}:
  0.20535
 -0.24415
  0.116354
```
"""
function Base.cross{T}(u::Vec{3, T}, v::Vec{3, T})
    @inbounds w = Vec{3, T}((u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]))
    return w
end
function Base.cross{T}(u::Vec{2, T}, v::Vec{2, T})
    @inbounds w = Vec{3, T}((zero(T), zero(T), u[1]*v[2] - u[2]*v[1]))
    return w
end
function Base.cross{T}(::Vec{1, T}, ::Vec{1, T})
    return zero(Vec{3,T})
end
