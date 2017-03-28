immutable PrimeDiff{N, F}
    f::F
end

@generated function (::Type{PrimeDiff{N}}){N, F}(f::F)
    ex = :(PrimeDiff{1, F}(f))
    for i in 2:N
        ex = :(_add_prime($ex))
    end
    return quote
        $(Expr(:meta, :inline))
        $ex
    end
end

@generated function _add_prime{N}(p::PrimeDiff{N})
    return quote
        $(Expr(:meta, :inline))
        PrimeDiff{$(N+1), typeof(p)}(p)
    end
end
@inline Base.ctranspose{N}(p::PrimeDiff{N}) = _add_prime(p)

@inline (p::PrimeDiff)(x) = gradient(p.f, x)

macro use_prime_diff()
    :(@inline Base.ctranspose{F <: Function}(f::F) = PrimeDiff{1}(f))
end

immutable By{N} end

@generated function (p::PrimeDiff{N}){N, M}(::Type{By{M}}, xs...)
    args = [i == M ? :x : :(xs[$i]) for i in 1:length(xs)]
    return quote
        $(Expr(:meta, :inline))
        g = x -> _original_function(p)($(args...))
        PrimeDiff{N}(g)(xs[M])
    end
end

@generated function _original_function{N}(p::PrimeDiff{N})
    ex = :(p.f)
    for i in 2:N
        ex = :($ex.f)
    end
    return quote
        $(Expr(:meta, :inline))
        $ex
    end
end
