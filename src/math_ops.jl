# norm, det, inv, eig, trace, dev
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

@gen_code function Base.norm{dim, T}(S::SymmetricTensor{4, dim, T})
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    @code :(data = get_data(S))
    @code :(s = zero(T))
    for k in 1:dim, l in 1:k, i in 1:dim, j in 1:i
        @code :(@inbounds v = data[$(idx(i,j,k,l))])
        if i == j && k == l
             @code :(s += v*v)
        elseif i == j || k == l
             @code :(s += 2*v*v)
        else
             @code :(s += 4*v*v)
        end
    end
    @code :(return sqrt(s))
end

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
    idx(i,j) = compute_index(get_base(t), i, j)
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
    idx(i,j) = compute_index(get_base(t), i, j)
    @code :($(Expr(:meta, :inline)))
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return get_base(typeof(t))((dinv,)))
    elseif dim == 2
        @code :(return get_base(typeof(t))((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                           -v[$(idx(1,2))] * dinv, v[$(idx(1,1))] * dinv)))
    else
        @code :(return get_base(typeof(t))((  (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
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

@gen_code function Base.inv{dim, T}(t::SymmetricTensor{2, dim, T})
    idx(i,j) = compute_index(get_base(t), i, j)
    @code :($(Expr(:meta, :inline)))
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return get_base(typeof(t))((dinv,)))
    elseif dim == 2
        @code :(return get_base(typeof(t))((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                            v[$(idx(1,1))] * dinv)))
    else
        @code :(return get_base(typeof(t))(( (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                                            -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                                             (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                                             (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                                            -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                                             (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv)))
    end
end

"""
Computes the eigenvalues and eigenvectors of a symmetric second order tensor.

```julia
eig(::SymmetricSecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2,3})
3×3 ContMechTensors.SymmetricTensor{2,3,Float64,6}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147

julia> Λ, Φ = eig(A);

julia> Λ
3-element ContMechTensors.Tensor{1,3,Float64,3}:
 -0.312033
  0.15636
  2.06075

julia> Φ
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
  0.492843   0.684993  0.536554
 -0.811724   0.139855  0.567049
  0.313385  -0.715     0.624952

julia> Φ ⋅ diagm(Tensor{2,3}, Λ) ⋅ inv(Φ) # Same as A
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.590845  0.766797  0.566237
 0.766797  0.460085  0.794026
 0.566237  0.794026  0.854147
```
"""
function Base.eig{dim, T, M}(S::SymmetricTensor{2, dim, T, M})
    S_m = convert(Tensor{2,dim}, S)
    λ, ϕ = eig(tomatrix(S_m))
    Λ = Tensor{1, dim}(λ)
    Φ = Tensor{2, dim}(ϕ)
    return Λ, Φ
end

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
    idx(i,j) = compute_index(get_base(S), i, j)
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
Computes the deviatoric part of a second order tensor.

```julia
dev(::SecondOrderTensor)
```

**Example:**

```jldoctest
julia> A = rand(Tensor{2,3});

julia> dev(A)
3×3 ContMechTensors.Tensor{2,3,Float64,9}:
 0.0469421  0.460085   0.200586
 0.766797   0.250123   0.298614
 0.566237   0.854147  -0.297065

julia> trace(dev(A))
0.0
```
"""
@generated function dev{dim, T, M}(S::Tensor{2, dim, T, M})
    f = (i,j) -> i == j ? :((get_data(S)[$(compute_index(Tensor{2, dim}, i, j))] - tr/3)) :
                           :(get_data(S)[$(compute_index(Tensor{2, dim}, i, j))])
    exp = tensor_create(Tensor{2, dim, T}, f)
    return quote
        $(Expr(:meta, :inline))
        tr = trace(S)
        Tensor{2, dim}($exp)
    end
end

@generated function dev{dim, T, M}(S::SymmetricTensor{2, dim, T, M})
    f = (i,j) -> i == j ? :((get_data(S)[$(compute_index(SymmetricTensor{2, dim}, i, j))] - tr/3)) :
                           :(get_data(S)[$(compute_index(SymmetricTensor{2, dim}, i, j))])
    exp = tensor_create(SymmetricTensor{2, dim, T},f)
    return quote
        $(Expr(:meta, :inline))
        tr = trace(S)
        SymmetricTensor{2, dim}($exp)
    end
end
