#############
# Promotion #
#############

# Promotion between two tensors promote the eltype and promotes
# symmetric tensors to tensors

function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{SymmetricTensor{order, dim, A, M}},
                                                                     ::Type{SymmetricTensor{order, dim, B, M}})
    SymmetricTensor{order, dim, promote_type(A, B), M}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{Tensor{order, dim, A, M}},
                                                                     ::Type{Tensor{order, dim, B, M}})
    Tensor{order, dim, promote_type(A, B), M}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{SymmetricTensor{order, dim, A, M1}},
                                                                          ::Type{Tensor{order, dim, B, M2}})
    Tensor{order, dim, promote_type(A, B), M2}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{Tensor{order, dim, A, M1}},
                                                                          ::Type{SymmetricTensor{order, dim, B, M2}})
    Tensor{order, dim, promote_type(A, B), M1}
end


###############
# Conversions #
###############

# Identity conversions
@inline Base.convert{order, dim, T}(::Type{Tensor{order, dim, T}}, t::Tensor{order, dim, T}) = t
@inline function Base.convert{order, dim, T1, T2, M}(::Type{Tensor{order, dim, T1, M}}, t::Tensor{order, dim, T2, M})
    Tensor{order, dim}(convert(NTuple{M, T1}, t.data))
end
@inline Base.convert{order, dim, T1, T2, M}(::Type{Tensor{order, dim, T1}}, t::Tensor{order, dim, T2, M}) = convert(Tensor{order, dim, T1, M}, t)


@inline Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, t::SymmetricTensor{order, dim, T}) = t
@inline function Base.convert{order, dim, T1, T2, M}(::Type{SymmetricTensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim, T2, M})
    SymmetricTensor{order, dim}(convert(NTuple{M, T1}, t.data))
end
@inline Base.convert{order, dim, T1, T2, M}(::Type{SymmetricTensor{order, dim, T1}}, t::SymmetricTensor{order, dim, T2, M}) = convert(SymmetricTensor{order, dim, T1, M}, t)

# Convert dimensions
@generated function Base.convert{order, dim1, dim2, T1}(::Type{Tensor{order, dim1}}, t::Tensor{order, dim2, T1})
    N = n_components(Tensor{order, dim1})
    exps = Expr[]
    if order == 1
         for i in 1:dim1
            if i > dim2
                push!(exps, :(zero(T1)))
            else
                push!(exps, :(t.data[$i]))
            end
        end
    end

    if order == 2
        for j in 1:dim1, i in 1:dim1
            if i > dim2 || j > dim2
                push!(exps, :(zero(T1)))
            else
                push!(exps, :(t.data[$(compute_index(Tensor{order, dim2}, i, j))]))
            end
        end
    end
    if order == 4
        for l in 1:dim1, k in 1:dim1, j in 1:dim1, i in 1:dim1
            if i > dim2 || j > dim2 || k > dim2  || l > dim2
                push!(exps, :(zero(T1)))
            else
                push!(exps, :(t.data[$(compute_index(Tensor{order, dim2}, i, j, k, l))]))
            end
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        v = $exp
        Tensor{order, dim1, T1, $N}(v)
    end
end

@generated function Base.convert{order, dim1, dim2, T1}(::Type{SymmetricTensor{order, dim1}}, t::SymmetricTensor{order, dim2, T1})
    N = n_components(SymmetricTensor{order, dim1})
    exps = Expr[]
    if order == 2
        for i in 1:dim1, j in i:dim1
            if i > dim2 || j > dim2
                push!(exps, :(zero(T1)))
            else
                push!(exps, :(t.data[$(compute_index(SymmetricTensor{order, dim2}, i, j))]))
            end
        end
    end
     if order == 4
        for k in 1:dim1, l in k:dim1, i in 1:dim1, j in i:dim1
            if i > dim2 || j > dim2 || k > dim2  || l > dim2
                push!(exps, :(zero(T1)))
            else
                push!(exps, :(t.data[$(compute_index(SymmetricTensor{order, dim2}, i, j, k, l))]))
            end
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
        $(Expr(:meta, :inline))
        v = $exp
        SymmetricTensor{order, dim1, T1, $N}(v)
    end
end


# Converting general data to a (symmetric) tensor. We leave the type of data unspecified to allow anything
# that fulfil the contract of having a getindex and length.
@generated function Base.convert{order, dim, T}(Tt::Union{Type{Tensor{order, dim, T}}, Type{SymmetricTensor{order, dim, T}}}, data)
    N = n_components(get_main_type(get_type(Tt)){order, dim})
    return quote
        @assert length(data) == $N
        Tv = promote_type(T, eltype(data))
        get_main_type(Tt){order,dim,Tv, $N}(to_tuple(NTuple{$N, Tv}, data))
    end
end

# Conversions to a type where the element type of the tensor is unspecified
# calls the conversions to a type where T = eltype(data)
function Base.convert{order, dim}(Tt::Union{Type{Tensor{order, dim}}, Type{SymmetricTensor{order, dim}}}, data)
    convert(get_main_type(Tt){order, dim, eltype(data)}, data)
end

# SymmetricTensor -> Tensor
# We unroll the creation by calling the compute_index function
@generated function Base.convert{order, dim, T1, T2, M1, M2}(::Type{Tensor{order, dim, T1, M1}}, t::SymmetricTensor{order, dim, T2, M2})
    N = n_components(Tensor{order, dim})
    rows = Int(N^(1/order))
    exps = Expr[]
    # Compute (row, col) from linear index
    if order == 2
        for j in 1:rows, i in 1:rows
            push!(exps, :(t.data[$(compute_index(SymmetricTensor{order, dim}, i, j))]))
        end
    else
        for l in 1:rows, k in 1:rows, j in 1:rows, i in 1:rows
            push!(exps, :(t.data[$(compute_index(SymmetricTensor{order, dim}, i, j, k, l))]))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            v = $exp
            Tensor{order, dim, promote_type(T1, T2), M1}(v)
        end
end

@generated function Base.convert{order, dim, T, M}(::Type{Tensor{order, dim}}, t::SymmetricTensor{order, dim, T, M})
    N = n_components(Tensor{order, dim})
    return quote
        $(Expr(:meta, :inline))
        convert(Tensor{order, dim, T, $N}, t)
    end
end

Base.convert{order, dim, T, M}(::Type{Tensor}, t::SymmetricTensor{order, dim, T, M}) = convert(Tensor{order, dim}, t)
Base.convert{order, dim, T, M}(::Type{SymmetricTensor}, t::Tensor{order, dim, T, M}) = convert(SymmetricTensor{order, dim}, t)


# Tensor -> SymmetricTensor
function issymmetric{dim}(t::Tensor{2, dim})
    N = n_components(Tensor{2, dim})
    rows = Int(N^(1/2))
    @inbounds for row in 1:rows, col in row:rows
        if t[row, col] != t[col, row]
            return false
        end
    end
    return true
end

function isminorsymmetric{dim}(t::Tensor{4, dim})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    @inbounds for k in 1:rows, l in k:rows, i in 1:rows, j in i:rows
        if t[i,j,k,l] != t[j,i,k,l] || t[i,j,k,l] != t[i,j,l,k]
            return false
        end
    end
    return true
end

isminorsymmetric(::SymmetricTensor{4}) = true

function ismajorsymmetric{dim}(t::FourthOrderTensor{dim})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    @inbounds for k in 1:rows, l in k:rows, i in 1:rows, j in i:rows
        if t[i,j,k,l] != t[k,l,i,j]
            return false
        end
    end
    return true
end

issymmetric(t::Tensor{4}) = isminorsymmetric(t)

issymmetric(::SymmetricTensors) = true

@generated function Base.convert{dim, T1, T2, M1, M2}(::Type{SymmetricTensor{2, dim, T1, M1}}, t::Tensor{2, dim, T2, M2})
    N = n_components(Tensor{2, dim})
    rows = Int(N^(1/2))
    exps = Expr[]
    for row in 1:rows, col in row:rows
        push!(exps, :(t.data[$(compute_index(Tensor{2, dim}, row, col))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            if issymmetric(t)
                return SymmetricTensor{2, dim, promote_type(T1, T2), M1}($exp)
            else
                throw(InexactError())
            end
        end
end

@generated function Base.convert{dim, T1, T2, M1, M2}(::Type{SymmetricTensor{4, dim, T1, M1}}, t::Tensor{4, dim, T2, M2})
    N = n_components(Tensor{4, dim})
    rows = Int(N^(1/4))
    exps = Expr[]
    for k in 1:rows, l in k:rows, i in 1:rows, j in i:rows
        push!(exps, :(t.data[$(compute_index(Tensor{4, dim}, i, j, k, l))]))
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            if issymmetric(t)
                return SymmetricTensor{4, dim, promote_type(T1, T2), M1}($exp)
            else
                throw(InexactError())
            end
        end
end

@generated function Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, t::Tensor{order, dim, T})
    N = n_components(SymmetricTensor{order, dim})
    return quote
        $(Expr(:meta, :inline))
        convert(SymmetricTensor{order, dim, T, $N}, t)
    end
end

@generated function Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim}}, t::Tensor{order, dim, T})
    N = n_components(SymmetricTensor{order, dim})
    return quote
        $(Expr(:meta, :inline))
        convert(SymmetricTensor{order, dim, T, $N}, t)
    end
end
