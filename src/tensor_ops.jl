######################
# Double contraction #
######################

@inline function dcontract{dim}(S1::Tensor{2, dim}, S2::Tensor{2, dim})
    return A_dot_B(S1.data, S2.data)
end

@inline function dcontract{dim, T1, T2, M}(S1::Tensor{4, dim, T1, M}, S2::Tensor{4, dim, T2})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{4, dim, Tv, M}(Am_mul_Bm(S1.data, S2.data))
end

@inline function dcontract{dim, T1, T2, M}(S1::Tensor{4, dim, T1}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{2, dim, Tv, M}(Am_mul_Bv(S1.data, S2.data))
end

@inline function dcontract{dim,T1,T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{4, dim, T2})
    Tv = typeof(zero(T1)*zero(T2))
    Tensor{2, dim, Tv, M}(Amt_mul_Bv(S2.data, S1.data))
end

const ⊡ = dcontract

# Promotion
dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dcontract(promote(S1, S2)...)
dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{2, dim}) = dcontract(S1, convert(Tensor, S2))
dcontract{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{4, dim}) = dcontract(S1, convert(Tensor, S2))

dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dcontract(promote(S1, S2)...)
dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{2, dim}) = dcontract(convert(Tensor, S1), S2)
dcontract{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{4, dim}) = dcontract(convert(Tensor, S1), S2)

dcontract{dim}(S1::Tensor{4, dim}, S2::SymmetricTensor{4, dim}) = dcontract(promote(S1, S2)...)
dcontract{dim}(S1::SymmetricTensor{4, dim}, S2::Tensor{4, dim}) = dcontract(promote(S1, S2)...)


########
# Norm #
########

"""
Computes the norm of a tensor
"""
@inline Base.norm(v::Vec) = sqrt(dot(v,v))
@inline Base.norm(S::SecondOrderTensor) = sqrt(dcontract(S, S))
@inline Base.norm(S::Tensor{4}) = sqrt(sumabs2(get_data(S)))


################
# Open product #
################

"""
Computes the outer product between two tensors `t1` and `t2`. Can also be called via the infix operator `⊗`.
"""

@generated function otimes{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    N = n_components(Tensor{4, dim})
    return quote
        $(Expr(:meta, :inline))
        Tv = typeof(zero(T1)*zero(T2))
       Tensor{4, dim, Tv, $N}(A_otimes_B(S1.data, S2.data))
    end
end

@generated function otimes{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2})
    N = n_components(Tensor{2, dim})
    return quote
        $(Expr(:meta, :inline))
        Tv = typeof(zero(T1)*zero(T2))
        Tensor{2, dim, Tv, $N}(A_otimes_B(v1.data, v2.data))
    end
end

const ⊗ = otimes

# Promotion
otimes{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = otimes(promote(S1, S2)...)
otimes{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = otimes(promote(S1, S2)...)


###############
# Dot product #
###############
@inline Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = A_dot_B(v1.data, v2.data)

@inline function Base.dot{dim, T1, T2}(S1::Tensor{2, dim, T1}, v2::Vec{dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Am_mul_Bv(S1.data, v2.data))
end

@inline function Base.dot{dim, T1, T2}(v1::Vec{dim, T1}, S2::Tensor{2, dim, T2})
    Tv = typeof(zero(T1) * zero(T2))
    return Vec{dim, Tv}(Amt_mul_Bv(S2.data, v1.data))
end

@inline function Base.dot{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1) * zero(T2))
    return Tensor{2, dim, Tv, M}(Am_mul_Bm(S1.data, S2.data))
end

#########################
# transpose-dot product #
#########################
@inline tdot{dim, T1, T2}(v1::Vec{dim, T1}, S2::SecondOrderTensor{dim, T2}) = dot(v1, S2)
@inline tdot{dim, T1, T2}(S1::SecondOrderTensor{dim, T1}, v2::Vec{dim, T2}) = dot(v2, S1)
@inline tdot{dim, T1, T2}(v1::Vec{dim, T1}, v2::Vec{dim, T2}) = dot(v1, v2)

@inline function tdot{dim, T1, T2, M}(S1::Tensor{2, dim, T1, M}, S2::Tensor{2, dim, T2, M})
    Tv = typeof(zero(T1) * zero(T2))
    return Tensor{2, dim, Tv, M}(Amt_mul_Bm(S1.data, S2.data))
end

@inline tdot{dim, T1, T2, M}(S1::SymmetricTensor{2, dim, T1, M}, S2::SymmetricTensor{2, dim, T2, M}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::SymmetricTensor{2, dim, T1, M1}, S2::Tensor{2, dim, T2, M2}) = dot(S1,S2)
@inline tdot{dim, T1, T2, M1, M2}(S1::Tensor{2, dim, T1, M1}, S2::SymmetricTensor{2, dim, T2, M2}) = tdot(promote(S1,S2)...)

@inline function Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::SymmetricTensor{2, dim})
    S1_t = convert(Tensor{2, dim}, S1)
    S2_t = convert(Tensor{2, dim}, S2)
    return Tensor{2, dim}(Am_mul_Bm(S1_t.data, S2_t.data))
end

@inline function tdot{dim}(S1::Tensor{2, dim})
    return SymmetricTensor{2, dim}(transpdot(S1.data))
end

@inline tdot{dim}(S1::SymmetricTensor{2,dim}) = tdot(convert(Tensor{2,dim}, S1))

# Promotion
Base.dot{dim}(S1::Tensor{2, dim}, S2::SymmetricTensor{2, dim}) = dot(promote(S1, S2)...)
Base.dot{dim}(S1::SymmetricTensor{2, dim}, S2::Tensor{2, dim}) = dot(promote(S1, S2)...)


#########
# Trace #
#########

import Base.LinAlg.trace
"""
Computes the trace of a second or fourth order tensor.
"""
@gen_code function trace{dim, T}(S::Union{SecondOrderTensor{dim, T}, FourthOrderTensor{dim, T}})
    @code :($(Expr(:meta, :inline)))
    @code :(s = zero(T))
    for i = 1:dim
        if S <: SecondOrderTensor
            @code :(@inbounds s += S[$i,$i])
        elseif S <: FourthOrderTensor
            @code :(@inbounds s += S[$i,$i,$i,$i])
        end
    end
    @code :(return s)
end


#######
# Vol #
#######

vol(S::SecondOrderTensor) = trace(S)


########
# Mean #
########

Base.mean(S::SecondOrderTensor) = trace(S) / 3


###############
# Determinant #
###############

"""
Computes the determinant of a second order tensor.
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


###########
# Inverse #
###########

import Base.inv

"""
Computes the inverse of a second order tensor.
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
    f = (i,j) -> i == j ? :((S.data[$(compute_index(Tensor{2, dim}, i, j))] - 1/3*tr)) :
                           :(S.data[$(compute_index(Tensor{2, dim}, i, j))])
    exp = tensor_create(Tensor{2, dim, T}, f)
    Tv = typeof(zero(T) * 1 / 3)
    return quote
        $(Expr(:meta, :inline))
        tr = trace(S)
        Tensor{2, dim, $Tv, M}($exp)
    end
end

###################
# Permute indices #
###################

function permute_index{dim}(S::FourthOrderTensor{dim},idx::NTuple{4,Int})
    sort([idx...]) == [1,2,3,4] || throw(ArgumentError("Missing index."))
    neworder = sortperm([idx...])
    f = (i,j,k,l) -> S[[i,j,k,l][neworder]...]
    return Tensor{4,dim}(f)
end


#############
# Transpose #
#############

"""
Computes the transpose of a tensor.
"""
@inline Base.transpose(S::Vec) = S

@inline function Base.transpose(S::Tensor{2})
    typeof(S)(mat_transpose(S.data))
end
Base.transpose(S::SymmetricTensor{2}) = S

"""
Computes the minor transpose of a fourth order tensor.
"""
@generated function minortranspose{dim, T, M}(t::Tensor{4, dim, T, M})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        push!(exps, :(t.data[$(compute_index(Tensor{4, dim}, j, i, l, k))]))
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
"""
@generated function majortranspose{dim, T}(t::FourthOrderTensor{dim, T})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        push!(exps, :(t.data[$(compute_index(get_base(t), k, l, i, j))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            Tensor{4, dim, T, $N}($exp)
        end
end

Base.ctranspose(S::AllTensors) = transpose(S)


############################
# Symmetric/Skew-symmetric #
############################

@inline symmetric(S1::SymmetricTensors) = S1

"""
Computes the symmetric part of a second order tensor, returns a `SymmetricTensor{2}`
"""
@generated function symmetric{dim, T}(t::Tensor{2, dim, T})
    N = n_components(SymmetricTensor{2, dim})
    rows = Int(div(sqrt(1 + 8*N), 2))
    exps = Expr[]
    for row in 1:rows, col in row:rows
        if row == col
            push!(exps, :(t.data[$(compute_index(Tensor{2, dim}, row, col))]))
        else
            I = compute_index(Tensor{2, dim}, row, col)
            J = compute_index(Tensor{2, dim}, col, row)
            push!(exps, :(0.5 * (t.data[$I] + t.data[$J])))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            SymmetricTensor{2, dim, T, $N}($exp)
        end
end

"""
Computes the (minor) symmetric part of a fourth order tensor, returns a `SymmetricTensor{4}`
"""
@generated function minorsymmetric{dim, T}(t::Tensor{4, dim, T})
    N = n_components(Tensor{4, dim})
    M = n_components(SymmetricTensor{4,dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for k in 1:rows, l in k:rows, i in 1:rows, j in i:rows
        if i == j && k == l
            push!(exps, :(t.data[$(compute_index(Tensor{4, dim}, i, j, k, l))]))
        else
            I = compute_index(Tensor{4, dim}, i, j, k, l)
            J = compute_index(Tensor{4, dim}, j, i, k, l)
            K = compute_index(Tensor{4, dim}, i, j, k, l)
            L = compute_index(Tensor{4, dim}, i, j, l, k)
            push!(exps, :(0.25 * (t.data[$I] + t.data[$J] + t.data[$K] + t.data[$L])))
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
"""
@generated function majorsymmetric{dim, T}(t::FourthOrderTensor{dim, T})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
        if i == j == k == l || i == k && j == l
            push!(exps, :(t.data[$(compute_index(get_base(t), i, j, k, l))]))
        else
            I = compute_index(get_base(t), i, j, k, l)
            J = compute_index(get_base(t), k, l, i, j)
            push!(exps, :(0.5 * (t.data[$I] + t.data[$J])))
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
"""
@inline skew(S1::Tensor{2}) = 0.5*(S1 - S1.')
@inline skew{dim,T}(S1::SymmetricTensor{2,dim,T}) = zero(Tensor{2,dim,T})


#######
# Eig #
#######

"""
Computes the eigenvalues and eigendirections of a second order tensor.
"""
function Base.eig{dim, T, M}(S::Tensor{2, dim, T, M})
    S_m = reshape(S[:], (dim, dim))
    λ, ϕ = eig(S_m)
    Λ = Tensor{1, dim}(λ)
    Φ = Tensor{2, dim}(ϕ)
    return Λ, Φ
end

#########
# Cross #
#########

"""
Computes the cross product between two `Vec` vectors, returns a `Vec{3}`. For dimensions 1 and 2 the `Vec`'s
are expanded to 3D first.
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
