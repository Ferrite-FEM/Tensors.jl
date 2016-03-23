#############
# Promotion #
#############

# Promotion between two tensors promote the eltype and promotes
# symmetric tensors to tensors

function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{SymmetricTensor{order, dim, A, M}},
                                                                     ::Type{SymmetricTensor{order, dim, B, M}})
    Base.@_pure_meta
    SymmetricTensor{order, dim, promote_type(A, B), M}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M}(::Type{Tensor{order, dim, A, M}},
                                                                     ::Type{Tensor{order, dim, B, M}})
    Base.@_pure_meta
    Tensor{order, dim, promote_type(A, B), M}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{SymmetricTensor{order, dim, A, M1}},
                                                                          ::Type{Tensor{order, dim, B, M2}})
    Base.@_pure_meta
    Tensor{order, dim, promote_type(A, B), M2}
end

function Base.promote_rule{dim , A <: Number, B <: Number, order, M1, M2}(::Type{Tensor{order, dim, A, M1}},
                                                                          ::Type{SymmetricTensor{order, dim, B, M2}})
    Base.@_pure_meta
    Tensor{order, dim, promote_type(A, B), M1}
end


###############
# Conversions #
###############

# Identity conversions
@inline Base.convert{order, dim, T}(::Type{Tensor{order, dim, T}}, t::Tensor{order, dim, T}) = t
@inline function Base.convert{order, dim, T1, T2, M}(::Type{Tensor{order, dim, T1, M}}, t::Tensor{order, dim, T2, M})
    Base.@_pure_meta
    Tensor{order, dim}(convert(NTuple{M, T1}, t.data))
end

@inline Base.convert{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, t::SymmetricTensor{order, dim, T}) = t
@inline function Base.convert{order, dim, T1, T2, M}(::Type{SymmetricTensor{order, dim, T1, M}}, t::SymmetricTensor{order, dim, T2, M})
    Base.@_pure_meta
    SymmetricTensor{order, dim}(convert(NTuple{M, T1}, t.data))
end

# Convert dimensions
@generated function Base.convert{order, dim1, dim2, T1}(::Type{Tensor{order, dim1}}, t::Tensor{order, dim2, T1})
    N = n_components(Tensor{order, dim1})
    exps = Expr[]
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

# Tensor -> SymmetricTensor
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
@generated function Base.convert{order, dim, T1, T2, M1, M2}(::Type{SymmetricTensor{order, dim, T1, M1}}, t::Tensor{order, dim, T2, M2})
    N = n_components(SymmetricTensor{order, dim})
    rows = Int(div(sqrt(1 + 8*N), 2))
    exps = Expr[]
    for row in 1:rows, col in row:rows
        if row == col
            push!(exps, :(t.data[$(compute_index(Tensor{order, dim}, row, col))]))
        else
            I = compute_index(Tensor{order, dim}, row, col)
            J = compute_index(Tensor{order, dim}, col, row)
            push!(exps, :(0.5 * (t.data[$I] + t.data[$J])))
        end
    end
    exp = Expr(:tuple, exps...)
    return quote
            $(Expr(:meta, :inline))
            SymmetricTensor{order, dim, promote_type(T1, T2), M1}($exp)
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
