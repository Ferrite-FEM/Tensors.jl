# symmetric, skew-symmetric and symmetric checks
"""
```julia
symmetric(::SecondOrderTensor)
symmetric(::FourthOrderTensor)
```
Computes the symmetric part of a second or fourth order tensor.
For a fourth order tensor, the symmetric part is the same as the minor symmetric part.
Returns a `SymmetricTensor`.

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
    exps = Expr[]
    for j in 1:dim, i in j:dim
        if i == j
            push!(exps, :(get_data(t)[$(compute_index(Tensor{2, dim}, i, j))]))
        else
            I = compute_index(Tensor{2, dim}, i, j)
            J = compute_index(Tensor{2, dim}, j, i)
            push!(exps, :((get_data(t)[$I] + get_data(t)[$J])/2))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        SymmetricTensor{2, dim}($exp)
    end
end

"""
```julia
minorsymmetric(::FourthOrderTensor)
```
Computes the minor symmetric part of a fourth order tensor, returns a `SymmetricTensor{4}`.
"""
@generated function minorsymmetric{dim}(t::Tensor{4, dim})
    exps = Expr[]
    for l in 1:dim, k in l:dim, j in 1:dim, i in j:dim
        if i == j && k == l
            push!(exps, :(data[$(compute_index(Tensor{4, dim}, i, j, k, l))]))
        else
            I = compute_index(Tensor{4, dim}, i, j, k, l)
            J = compute_index(Tensor{4, dim}, j, i, k, l)
            K = compute_index(Tensor{4, dim}, i, j, k, l)
            L = compute_index(Tensor{4, dim}, i, j, l, k)
            push!(exps, :((data[$I] + data[$J] + data[$K] + data[$L]) / 4))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        data = get_data(t)
        SymmetricTensor{4, dim}($exp)
    end
end

@inline minorsymmetric(t::SymmetricTensors) = t

@inline symmetric(t::Tensor{4}) = minorsymmetric(t)

"""
```julia
majorsymmetric(::FourthOrderTensor)
```
Computes the major symmetric part of a fourth order tensor, returns a `Tensor{4}`.
"""
@generated function majorsymmetric{dim}(t::FourthOrderTensor{dim})
    exps = Expr[]
    for l in 1:dim, k in 1:dim, j in 1:dim, i in 1:dim
        if i == j == k == l || i == k && j == l
            push!(exps, :(data[$(compute_index(get_base(t), i, j, k, l))]))
        else
            I = compute_index(get_base(t), i, j, k, l)
            J = compute_index(get_base(t), k, l, i, j)
            push!(exps, :((data[$I] + data[$J]) / 2))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        data = get_data(t)
        Tensor{4, dim}($exp)
    end
end

"""
```julia
skew(::SecondOrderTensor)
```
Computes the skew-symmetric (anti-symmetric) part of a second order tensor, returns a `Tensor{2}`.
"""
@inline skew(S1::Tensor{2}) = (S1 - S1.') / 2
@inline skew{dim,T}(S1::SymmetricTensor{2,dim,T}) = zero(Tensor{2,dim,T})

# Symmetry checks
@inline Base.issymmetric(t::Tensor{2, 1}) = true
@inline function Base.issymmetric(t::Tensor{2, 2})
    data = get_data(t)
    @inbounds return data[2] == data[3]
end
@inline function Base.issymmetric(t::Tensor{2, 3})
    data = get_data(t)
    @inbounds begin
        return (data[2] == data[4] &&
                data[3] == data[7] &&
                data[6] == data[8])
    end
end

function isminorsymmetric{dim}(t::Tensor{4, dim})
    @inbounds for l in 1:dim, k in l:dim, j in 1:dim, i in j:dim
        if t[i,j,k,l] != t[j,i,k,l] || t[i,j,k,l] != t[i,j,l,k]
            return false
        end
    end
    return true
end

isminorsymmetric(::SymmetricTensor{4}) = true

function ismajorsymmetric{dim}(t::FourthOrderTensor{dim})
    @inbounds for l in 1:dim, k in l:dim, j in 1:dim, i in j:dim
        if t[i,j,k,l] != t[k,l,i,j]
            return false
        end
    end
    return true
end

Base.issymmetric(t::Tensor{4}) = isminorsymmetric(t)

Base.issymmetric(::SymmetricTensors) = true
