import ForwardDiff: Dual, partials, value

@inline function Base.muladd{N, T <: SIMDTypes}(x::Dual{N, T}, y::Dual{N, T}, z::Dual{N, T})
    xp = tosimd(partials(x).values)
    yp = tosimd(partials(y).values)
    zp = tosimd(partials(z).values)
    parts = muladd(value(x), yp, muladd(value(y), xp, zp))
    Dual(muladd(value(x), value(y), value(z)), totuple(parts))
end

@inline function Base.muladd{N, T <: SIMDTypes}(x::Dual{N, T}, y::T, z::Dual{N, T})
    xp = tosimd(partials(x).values)
    zp = tosimd(partials(z).values)
    parts = muladd(xp, value(y), zp)
    Dual(muladd(value(x), value(y), value(z)), totuple(parts))
end

########
# SIMD #
########

@generated function totuple{N, T}(a::SIMD.Vec{N, T})
    ex = ForwardDiff.tupexpr(i -> :(a[$i]), N)
    return quote
        $(Expr(:meta, :inline))
        $ex
    end
end

@inline function ForwardDiff.scale_tuple{N, T <: SIMDTypes}(tup::NTuple{N, T}, x)
    return totuple(tosimd(tup) * x)
end

@inline function ForwardDiff.div_tuple_by_scalar{N, T <: SIMDTypes}(tup::NTuple{N, T}, x)
    return totuple(tosimd(tup) / x)
end

@inline function ForwardDiff.add_tuples{N, T <: SIMDTypes}(a::NTuple{N, T}, b::NTuple{N, T})
    return totuple(tosimd(a) + tosimd(b))
end

@inline function ForwardDiff.sub_tuples{N, T <: SIMDTypes}(a::NTuple{N, T}, b::NTuple{N, T})
    return totuple(tosimd(a) - tosimd(b))
end

@inline function ForwardDiff.minus_tuple{N, T <: SIMDTypes}(tup::NTuple{N, T})
    return totuple(-tosimd(tup))
end

@inline function ForwardDiff.mul_tuples{N, T <: SIMDTypes}(a::NTuple{N, T}, b::NTuple{N, T}, afactor, bfactor)
    return totuple( afactor * tosimd(a) + bfactor * tosimd(b) )
end
