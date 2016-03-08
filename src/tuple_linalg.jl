# This file contrains tuple operators that does linear algebra assuming
# that the tuple is a square matrix


@generated function mat_get_index{N}(t::NTuple{N}, i::Int, j::Int)
    rows = Int(sqrt(N))
    return quote
        $(Expr(:meta, :inline))
        @inbounds v = t[(j-1) * $rows + i]
        return v
    end
end

# Dot product between row, col
@generated function dot_matmat{N, T1, T2, transp}(A::NTuple{N, T1}, B::NTuple{N, T2}, i, j, ::Type{Val{transp}})
    rows = Int(sqrt(N))
    if !transp
        return quote
            $(Expr(:meta, :inline))
            s = zero(T1)*zero(T2)
            @inbounds for k = 1:$rows
                s += mat_get_index(A, i, k) * mat_get_index(B, k, j)
            end
            s
        end
    else
        return quote
            $(Expr(:meta, :inline))
            s = zero(T1)*zero(T2)
            @inbounds for k = 1:$rows
                s += mat_get_index(A, k, i) * mat_get_index(B, k, j)
            end
            s
        end
    end
end

# Dot product between row, vec
@gen_code function dot_matvec{N, M, T1, T2, transp}(A::NTuple{N, T1}, b::NTuple{M, T2}, i, ::Type{Val{transp}})
    rows = Int(sqrt(N))
    if !transp
        return quote
            $(Expr(:meta, :inline))
            s = zero(T1)*zero(T2)
            @inbounds for k = 1:$rows;
                s += mat_get_index(A, i, k) * b[k];
            end
            return s
        end
    else
         return quote
            $(Expr(:meta, :inline))
            s = zero(T1)*zero(T2)
            @inbounds for k = 1:$rows;
                s += mat_get_index(A, k, i) * b[k];
            end
            return s
        end
    end
end


@generated function Am_mul_Bm{N}(A::NTuple{N}, B::NTuple{N})
    rows = Int(sqrt(N))
    body = Expr(:tuple, [:(dot_matmat(A, B, $i, $j, Val{false})) for i=1:rows, j=1:rows]...)
    return quote
        $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end

@generated function Amt_mul_Bm{N}(A::NTuple{N}, B::NTuple{N})
    rows = Int(sqrt(N))
    body = Expr(:tuple, [:(dot_matmat(A, B, $i, $j, Val{true})) for i=1:rows, j=1:rows]...)
    return quote
        $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end

@generated function Am_mul_Bv{M, N}(A::NTuple{N}, b::NTuple{M})
    @assert sqrt(N) == M
    rows = Int(sqrt(N))
    body = Expr(:tuple, [:(dot_matvec(A, b, $i, Val{false})) for i=1:rows]...)
    return quote
        $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end

@generated function Amt_mul_Bv{M, N}(A::NTuple{N}, b::NTuple{M})
    @assert sqrt(N) == M
    rows = Int(sqrt(N))
    body = Expr(:tuple, [:(dot_matvec(A, b, $i, Val{true})) for i=1:rows]...)
    return quote
        $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end

@generated function A_dot_B{N, T1, T2}(a::NTuple{N, T1}, b::NTuple{N, T2})
    body = Expr(:block, [:(@inbounds s += (a[$i]*b[$i])) for i=1:N]...)
    return quote
        $(Expr(:meta, :inline))
       s = zero(T1) * zero(T2)
       $body
       return s
    end
end

@generated function A_otimes_B{N}(a::NTuple{N}, b::NTuple{N})
    body = Expr(:tuple, [:(a[$i]*b[$j]) for i=1:N, j = 1:N]...)
    return quote
        $(Expr(:meta, :inline))
       @inbounds d = $body
       return d
    end
end


function tupexpr_mat(f,N)
    ex = Expr(:tuple, [f(i,j) for i=1:Int(sqrt(N)), j = 1:Int(sqrt(N))]...)
    return quote
        @inbounds return $ex
    end
end



@generated function mat_set_index{N, T, I, J}(a::NTuple{N, T}, v, ::Type{Val{I}}, ::Type{Val{J}})
    rows = Int(sqrt(N))
    b = tupexpr_mat((i,j) -> (i == I && j == J) ? :(v) : :(a[$((j-1) * rows + i)]), N)
    return quote
        $(inline_body(T))
        $b
    end
end

@generated function vec_set_index{N, T, I}(a::NTuple{N, T}, v, ::Type{Val{I}})
    b = tupexpr((i) -> (i == I) ? :(v) : :(a[$(i)]), N)
    return quote
        $(inline_body(T))
        $b
    end
end

@generated function mat_transpose{N, T}(a::NTuple{N, T})
    rows = Int(sqrt(N))
    b = tupexpr_mat((i,j) -> :(a[$((i-1) * rows + j)]), N)
    return quote
        $(inline_body(T))
        $b
    end
end


#N = 3
#Atup = ((rand(N*N)...))
#Btup = ((rand(N*N)...))
#btup = ((rand(N)...))
##
#Am_mult_Bv(Atup, btup) == ( (reshape([Atup...], N, N) * reshape([btup...], N))...)
#Amt_mult_Bv(Atup, btup) == ( (reshape([Atup...], N, N)' * reshape([btup...], N))...)
#Am_mult_Bm(Atup, Btup) == ( (reshape([Atup...], N, N) * reshape([Btup...], N, N))...)
