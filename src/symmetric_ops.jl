#################################################
# Specialized Second Order Symmetric Operations #
#################################################

######################
# Double contraction #
######################

@gen_code function dcontract{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, S2::SymmetricTensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    @code :($(Expr(:meta, :inline)))
    @code :(s = zero($Tv);
            data1 = get_data(S1);
            data2 = get_data(S2))
     for k in 1:n_independent_components(dim, true)
        if is_diagonal_index(dim, k)
            @code :(@inbounds s += data1[$k] * data2[$k])
        else
            @code :(@inbounds s += 2 * data1[$k] * data2[$k])
        end
    end
    @code :(return s)
end

@generated function dcontract{dim, T1, T2, M}(S1::SymmetricTensor{2, dim, T1, M}, S2::SymmetricTensor{4, dim, T2})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
            for k in 1:dim, l in k:dim
            if k == l
                push!(exps_ele.args, :(data4[$(idx4(l, k, j, i))] * data2[$(idx2(l,k))]))
            else
                push!(exps_ele.args, :( 2 * data4[$(idx4(l, k, j, i))] * data2[$(idx2(l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    Tv = typeof(zero(T1) * zero(T2))
    quote
         data2 = get_data(S1)
         data4 = get_data(S2)
         @inbounds r = $exps
         SymmetricTensor{2, dim, $Tv, M}(r)
    end
end


@generated function dcontract{dim, T1, T2, M}(S1::SymmetricTensor{4, dim, T1}, S2::SymmetricTensor{2, dim, T2, M})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
            for k in 1:dim, l in k:dim
            if k == l
                push!(exps_ele.args, :(data4[$(idx4(j, i, l, k))] * data2[$(idx2(l,k))]))
            else
                push!(exps_ele.args, :( 2 * data4[$(idx4(j, i, l, k))] * data2[$(idx2(l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    Tv = typeof(zero(T1) * zero(T2))
    quote
         data2 = get_data(S2)
         data4 = get_data(S1)
         @inbounds r = $exps
         SymmetricTensor{2, dim, $Tv, M}(r)
    end
end

@generated function dcontract{dim, T1, T2, M}(S1::SymmetricTensor{4, dim, T1, M}, S2::SymmetricTensor{4, dim, T2, M})
    idx4(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    idx2(k,l) = compute_index(SymmetricTensor{2, dim}, k, l)
    exps = Expr(:tuple)
    for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
        for m in 1:dim, n in m:dim
            if m == n
                push!(exps_ele.args, :(data1[$(idx4(j,i,n,m))] * data2[$(idx4(m,n,l,k))]))
            else
                 push!(exps_ele.args, :(2*data1[$(idx4(j,i,n,m))] * data2[$(idx4(m,n,l,k))]))
            end
        end
        push!(exps.args, exps_ele)
    end
    Tv = typeof(zero(T1) * zero(T2))
    quote
         data2 = get_data(S2)
         data1 = get_data(S1)
         @inbounds r = $exps
         SymmetricTensor{4, dim, $Tv, M}(r)
    end
end


#######
# Dot #
#######

@generated function Base.dot{dim, T1, T2}(S1::SymmetricTensor{2, dim, T1}, v2::Vec{dim, T2})
    idx(i,j) = compute_index(SymmetricTensor{2, dim}, i, j)
    exps = Expr(:tuple)
    for i in 1:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
            for j in 1:dim
                push!(exps_ele.args, :(get_data(S1)[$(idx(i, j))] * get_data(v2)[$j]))
            end
        push!(exps.args, exps_ele)
    end
    Tv = typeof(zero(T1) * zero(T2))
    quote
         $(Expr(:meta, :inline))
         @inbounds r = $exps
         Vec{dim,$Tv}(r)
    end
end
@inline Base.dot{dim, T}(v2::Vec{dim, T}, S1::SymmetricTensor{2, dim, T}) = dot(S1, v2)


###########
# Inverse #
###########
@gen_code function Base.inv{dim, T}(t::SymmetricTensor{2, dim, T})
    idx(i,j) = compute_index(get_lower_order_tensor(t), i, j)
    @code :($(Expr(:meta, :inline)))
    @code :(dinv = 1 / det(t))
    @code :(v = get_data(t))
    if dim == 1
        @code :(return  typeof(t)((dinv,)))
    elseif dim == 2
        @code :( return typeof(t)((v[$(idx(2,2))] * dinv, -v[$(idx(2,1))] * dinv,
                                   v[$(idx(1,1))] * dinv)))
    else
        @code :(return typeof(t)((  (v[$(idx(2,2))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,2))]) * dinv,
                                   -(v[$(idx(2,1))]*v[$(idx(3,3))] - v[$(idx(2,3))]*v[$(idx(3,1))]) * dinv,
                                    (v[$(idx(2,1))]*v[$(idx(3,2))] - v[$(idx(2,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,1))]*v[$(idx(3,3))] - v[$(idx(1,3))]*v[$(idx(3,1))]) * dinv,
                                   -(v[$(idx(1,1))]*v[$(idx(3,2))] - v[$(idx(1,2))]*v[$(idx(3,1))]) * dinv,

                                    (v[$(idx(1,1))]*v[$(idx(2,2))] - v[$(idx(1,2))]*v[$(idx(2,1))]) * dinv)))
    end
end


########
# Norm #
########

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


#######
# Dev #
#######

@generated function dev{dim, T, M}(S::SymmetricTensor{2, dim, T, M})
    f = (i,j) -> i == j ? :((get_data(S)[$(compute_index(SymmetricTensor{2, dim}, i, j))] - tr/3)) :
                           :(get_data(S)[$(compute_index(SymmetricTensor{2, dim}, i, j))])
    exp = tensor_create(SymmetricTensor{2, dim, T},f)
    Tv = typeof(zero(T) /3)
    return quote
        $(Expr(:meta, :inline))
        tr = trace(S)
        SymmetricTensor{2, dim, $Tv, M}($exp)
    end
end


################
# Open product #
################

@inline function otimes{dim, T1, T2, M}(S1::SymmetricTensor{2, dim, T1, M}, S2::SymmetricTensor{2, dim, T2, M})
    N = n_components(SymmetricTensor{4, dim})
    Tv = typeof(zero(T1) * zero(T2))
    SymmetricTensor{4, dim, Tv, N}(tovector(S1) * tovector(S2)')
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

julia> eig(A)
([-0.312033,0.15636,2.06075],
[0.492843 -0.684993 -0.536554; -0.811724 -0.139855 -0.567049; 0.313385 0.715 -0.624952])
```
"""
function Base.eig{dim, T, M}(S::SymmetricTensor{2, dim, T, M})
    S_m = Symmetric(reshape(S[:], (dim, dim)))
    λ, ϕ = eig(S_m)
    Λ = Tensor{1, dim}(λ)
    Φ = Tensor{2, dim}(ϕ)
    return Λ, Φ
end

"""
Computes a special dot product between two vectors and a symmetric fourth order tensor
such that ``a_k C_{ikjl} b_l``.

```julia
dotdot(::Vec, ::SymmetricFourthOrderTensor, ::Vec)
```
"""
@generated function dotdot{dim, T1, T2, T3}(v1::Vec{dim, T1}, S::SymmetricTensor{4, dim, T2}, v2::Vec{dim, T3})
    idx(i,j,k,l) = compute_index(SymmetricTensor{4, dim}, i, j, k, l)
    N = n_components(Tensor{2, dim})
    Tv = typeof(one(T1) * one(T2) * one(T3))
    exps = Expr(:tuple)
    for j in 1:dim, i in 1:dim
        exps_ele = Expr(:call)
        push!(exps_ele.args, :+)
        for l in 1:dim, k in 1:dim
            push!(exps_ele.args, :(get_data(v1)[$k] * get_data(S)[$(idx(i,k,j,l))] * get_data(v2)[$l]))
        end
        push!(exps.args, exps_ele)
    end
    return quote
        $(Expr(:meta, :inline))
        @inbounds r = $exps
        Tensor{2, dim, $Tv, $N}(r)
    end
end
