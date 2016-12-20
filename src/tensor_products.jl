# dcontract, dot, tdot, otimes, cross
"""
```julia
dcontract(::SecondOrderTensor, ::SecondOrderTensor)
dcontract(::SecondOrderTensor, ::FourthOrderTensor)
dcontract(::FourthOrderTensor, ::SecondOrderTensor)
dcontract(::FourthOrderTensor, ::FourthOrderTensor)
```
Computes the double contraction between two tensors.
The symbol `⊡`, written `\\boxdot`, is overloaded for double contraction.
The reason `:` is not used is because it does not have the same precedence as multiplication.

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

@inline function dcontract{dim}(S1::Tensor{4, dim}, S2::Tensor{4, dim})
    Tensor{4, dim}(tomatrix(S1) * tomatrix(S2))
end

@inline function dcontract{dim}(S1::Tensor{4, dim}, S2::Tensor{2, dim})
    Tensor{2, dim}(tomatrix(S1) * tovector(S2))
end

@inline function dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{4, dim})
    Tensor{2, dim}(tomatrix(S2)' * tovector(S1))
end

const ⊡ = dcontract

# Specialized methods for symmetric tensors
@generated function dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    idx2(i,j) = compute_index(SymmetricTensor{2, dim}, i, j)
    ex = Expr[]
    for j in 1:dim, i in j:dim
        if i == j
            push!(ex, :(get_data(S1)[$(idx2(i, j))] * get_data(S2)[$(idx2(i, j))]))
        else
            push!(ex, :(2 * get_data(S1)[$(idx2(i, j))] * get_data(S2)[$(idx2(i, j))]))
        end
    end
    exp = reduce((ex1,ex2) -> :(+($ex1, $ex2)), ex)
    return quote
        $(Expr(:meta, :inline))
        $exp
    end
end

@generated function dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{4, dim})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(i,j) = compute_index(Tensor{2, dim}, i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in l:dim
        exps_ele = Expr[]
        for j in 1:dim, i in 1:dim
            push!(exps_ele, :(data2[$(idx2(i, j))] * data4[$(idx4(i, j, k, l))]))
        end
        push!(exps.args, reduce((ex1,ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    quote
        $(Expr(:meta, :inline))
        data2 = get_data(S1)
        data4 = get_data(S2)
        @inbounds r = $exps
        SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{2, dim})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(Tensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for j in 1:dim, i in j:dim
        exps_ele = Expr[]
        for l in 1:dim, k in 1:dim
            push!(exps_ele, :(data4[$(idx4(i, j, k, l))] * data2[$(idx2(k, l))]))
        end
        push!(exps.args, reduce((ex1,ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    quote
        $(Expr(:meta, :inline))
        data2 = get_data(S2)
        data4 = get_data(S1)
        @inbounds r = $exps
        SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{4, dim})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(i,j) = compute_index(SymmetricTensor{2, dim}, i, j)
    exps = Expr(:tuple)
    for l in 1:dim, k in l:dim
        exps_ele = Expr[]
        for j in 1:dim, i in j:dim
            if i == j
                push!(exps_ele, :(data2[$(idx2(i, j))] * data4[$(idx4(i, j, k, l))]))
            else
                push!(exps_ele, :(2 * data2[$(idx2(i, j))] * data4[$(idx4(i, j, k, l))]))
            end
        end
        push!(exps.args, reduce((ex1,ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    quote
        $(Expr(:meta, :inline))
        data2 = get_data(S1)
        data4 = get_data(S2)
        @inbounds r = $exps
        SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::SymmetricTensor{2, dim})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for j in 1:dim, i in j:dim
        exps_ele = Expr[]
        for l in 1:dim, k in l:dim
            if k == l
                push!(exps_ele, :(data4[$(idx4(i, j, k, l))] * data2[$(idx2(k, l))]))
            else
                push!(exps_ele, :(2 * data4[$(idx4(i, j, k, l))] * data2[$(idx2(k, l))]))
            end
        end
        push!(exps.args, reduce((ex1,ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    quote
        $(Expr(:meta, :inline))
        data2 = get_data(S2)
        data4 = get_data(S1)
        @inbounds r = $exps
        SymmetricTensor{2, dim}(r)
    end
end

@generated function dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::SymmetricTensor{4, dim})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    exps = Expr(:tuple)
    for l in 1:dim, k in l:dim, j in 1:dim, i in j:dim
        exps_ele = Expr[]
        for n in 1:dim, m in n:dim
            if m == n
                push!(exps_ele, :(data1[$(idx4(i, j, m, n))] * data2[$(idx4(m, n, k, l))]))
            else
                push!(exps_ele, :(2 * data1[$(idx4(i, j, m, n))] * data2[$(idx4(m, n, k, l))]))
            end
        end
        push!(exps.args, reduce((ex1,ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    quote
        $(Expr(:meta, :inline))
        data2 = get_data(S2)
        data1 = get_data(S1)
        @inbounds r = $exps
        SymmetricTensor{4, dim}(r)
    end
end

# Promotion
@inline dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{2, dim}) = dcontract(S1, convert(Tensor, S2))

@inline dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{4, dim}) = dcontract(convert(Tensor, S1), S2)

@inline dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{4, dim}) = dcontract(promote(S1, S2)...)
@inline dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{4, dim}) = dcontract(promote(S1, S2)...)

"""
```julia
otimes(::Vec, ::Vec)
otimes(::SecondOrderTensor, ::SecondOrderTensor)
```
Computes the open product between two tensors.
The symbol `⊗`, written `\\otimes`, is overloaded for tensor products.

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
@inline function otimes{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    Tensor{4, dim}(tovector(S1) * tovector(S2)')
end

@inline function otimes{dim}(v1::Vec{dim}, v2::Vec{dim})
    Tensor{2, dim}(tovector(v1) * tovector(v2)')
end

const ⊗ = otimes

# Specialized methods for symmetric tensors
@inline function otimes{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    SymmetricTensor{4, dim}(tovector(S1) * tovector(S2)')
end

# Promotion
@inline otimes{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = otimes(promote(S1, S2)...)
@inline otimes{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = otimes(promote(S1, S2)...)

"""
```julia
dot(::Vec, ::Vec)
dot(::Vec, ::SecondOrderTensor)
dot(::SecondOrderTensor, ::Vec)
dot(::SecondOrderTensor, ::SecondOrderTensor)
```
Computes the dot product (single contraction) between two tensors.
The symbol `⋅`, written `\\cdot`, is overloaded for single contraction.

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
@inline Base.dot{dim}(v1::Vec{dim}, v2::Vec{dim}) = tovector(v1) ⋅ tovector(v2)

@inline function Base.dot{dim}(S1::Tensor{2, dim}, v2::Vec{dim})
    return Vec{dim}(tomatrix(S1) * tovector(v2))
end

@inline function Base.dot{dim}(v1::Vec{dim}, S2::Tensor{2, dim})
    return Vec{dim}(tomatrix(S2)' * tovector(v1))
end

@inline function Base.dot{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return Tensor{2, dim}(tomatrix(S1) * tomatrix(S2))
end

@inline function Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    S1_t = convert(Tensor{2, dim}, S1)
    S2_t = convert(Tensor{2, dim}, S2)
    return Tensor{2, dim}(tomatrix(S1_t) * tomatrix(S2_t))
end

@inline Base.dot{dim}(S1::SymmetricTensor{2, dim}, v2::Vec{dim}) = dot(convert(Tensor{2, dim}, S1), v2)

@inline Base.dot{dim}(v2::Vec{dim}, S1::SymmetricTensor{2, dim}) = dot(S1, v2)

# Promotion
Base.dot{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dot(promote(S1, S2)...)
Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dot(promote(S1, S2)...)

"""
```julia
tdot(::Vec, ::Vec)
tdot(::Vec, ::SecondOrderTensor)
tdot(::SecondOrderTensor, ::Vec)
tdot(::SecondOrderTensor, ::SecondOrderTensor)
```
Computes the transpose-dot product (single contraction) between two tensors.

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
    return Tensor{2, dim}(tomatrix(S1)' * tomatrix(S2))
end

@inline tdot{dim, T1, T2, M}(S1::SymmetricTensor{2, dim, T1, M}, S2::SymmetricTensor{2, dim, T2, M}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::SymmetricTensor{2, dim, T1, M1}, S2::Tensor{2, dim, T2, M2}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::Tensor{2, dim, T1, M1}, S2::SymmetricTensor{2, dim, T2, M2}) = tdot(promote(S1,S2)...)

"""
```julia
tdot(::SecondOrderTensor)
```
Computes the transpose-dot of a second order tensor with itself.
Returns a `SymmetricTensor`.

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
@generated function tdot{dim}(S1::Tensor{2, dim})
    idx(i,j) = compute_index(Tensor{2, dim}, i, j)
    ex = Expr(:tuple)
    for j in 1:dim, i in j:dim
        exps_ele = Expr[]
        for k in 1:dim
            push!(exps_ele, :(get_data(S1)[$(idx(k,i))] * get_data(S1)[$(idx(k,j))]))
        end
        push!(ex.args, reduce((ex1, ex2) -> :(+($ex1, $ex2)), exps_ele))
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds r = $ex
        SymmetricTensor{2, dim}(r)
    end
end

@inline tdot{dim}(S1::SymmetricTensor{2,dim}) = tdot(convert(Tensor{2,dim}, S1))

"""
```julia
cross(::Vec, ::Vec)
```
Computes the cross product between two `Vec` vectors, returns a `Vec{3}`. For dimensions 1 and 2 the `Vec`'s
are expanded to 3D first. The infix operator × (written `\\times`) can also be used.

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
    @inbounds w = Vec{3}((u[2]*v[3] - u[3]*v[2], u[3]*v[1] - u[1]*v[3], u[1]*v[2] - u[2]*v[1]))
    return w
end
function Base.cross{T}(u::Vec{2, T}, v::Vec{2, T})
    @inbounds w = Vec{3}((zero(T), zero(T), u[1]*v[2] - u[2]*v[1]))
    return w
end
function Base.cross{T}(::Vec{1, T}, ::Vec{1, T})
    return zero(Vec{3,T})
end
