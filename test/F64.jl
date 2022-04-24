# dummy type wrapping a Float64 used in tests
struct F64 <: AbstractFloat
    x::Float64
end
F64(x::F64) = x

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

using Random
Random.rand(rng::AbstractRNG, ::Random.SamplerTrivial{Random.CloseOpen01{F64}}) = F64(rand(rng, Float64))
Random.randn(rng::Random.AbstractRNG, ::Type{F64}) = F64(randn(rng, Float64))

Base.sqrt(a::F64) = F64(sqrt(a.x))

# comparison
Base.isapprox(a::F64, b::F64) = isapprox(a.x, b.x)
Base.:<(a::F64, b::F64) = a.x < b.x
Base.:<=(a::F64, b::F64) = a.x <= b.x
Base.eps(::Type{F64}) = eps(Float64)

# promotion
Base.promote_type(::Type{Float32}, ::Type{F64}) = Float64 # for eig
Base.promote_type(::Type{Float64}, ::Type{F64}) = Float64 # for vecnorm
Base.promote_type(::Type{Int64}, ::Type{F64}) = F64 # for eig in StaticArrays
Base.promote_type(::Type{F64}, ::Type{Int64}) = F64 # for eig in StaticArrays
Base.promote(a::F64, b::T) where {T <: Number} = a, F64(b)
Base.promote(a::T, b::F64) where {T <: Number} = F64(a), b
Base.convert(::Type{F64}, a::F64) = a
Base.convert(::Type{Float64}, a::F64) = a.x
Base.convert(::Type{F64}, a::T) where {T <: Number} = F64(a)

# for testing of eigen
Base.acos(a::F64) = F64(acos(a.x))
Base.cos(a::F64) = F64(cos(a.x))
Base.sin(a::F64) = F64(sin(a.x))
Base.precision(::Type{F64}) = precision(Float64)
Base.floatmin(::Type{F64}) = floatmin(Float64)

# Number type which is not <: Real (Tensors#154)
struct NotReal <: Number
    x::Float64
end
