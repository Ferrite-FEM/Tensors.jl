# dummy type wrapping a Float64 used in tests
immutable F64 <: AbstractFloat
    x::Float64
end

# operations
for op in (:+, :-)
    @eval Base.$op(a::F64) = F64($op(a.x))
end
for op in (:+, :-, :*, :/)
    @eval Base.$op(a::F64, b::F64) = F64($op(a.x, b.x))
end
for op in(:zero, :one)
    @eval Base.$op(::Type{F64}) = F64($op(Float64))
end
Base.rand(rng::AbstractRNG, F64) = F64(rand())
Base.sqrt(a::F64) = F64(sqrt(a.x))

# comparison
Base.isapprox(a::F64, b::F64) = isapprox(a.x, b.x)
Base.:<(a::F64, b::F64) = a.x < b.x
Base.:<=(a::F64, b::F64) = a.x <= b.x
Base.eps(::Type{F64}) = eps(Float64)

# promotion
Base.promote_type(::Type{Float32}, ::Type{F64}) = Float64 # for eig
Base.promote_type(::Type{Float64}, ::Type{F64}) = Float64 # for vecnorm
Base.promote{T <: Number}(a::F64, b::T) = a, F64(b)
Base.promote{T <: Number}(a::T, b::F64) = F64(a), b
Base.convert(::Type{F64}, a::F64) = a
Base.convert(::Type{Float64}, a::F64) = a.x
Base.convert{T <: Number}(::Type{F64}, a::T) = F64(a)
