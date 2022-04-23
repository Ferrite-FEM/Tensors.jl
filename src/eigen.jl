# Specify conversion to static arrays for 2nd order tensors
to_smatrix(a::Tensor{2, dim, T}) where {dim, T} = SMatrix{dim, dim, T}(a)
to_smatrix(a::SymmetricTensor{2, dim, T}) where {dim, T} = Symmetric(SMatrix{dim, dim, T}(a))


"""
    eigvals(::SymmetricTensor)

Compute the eigenvalues of a symmetric tensor.
"""
@inline LinearAlgebra.eigvals(S::SymmetricTensor{4}) = eigvals(eigen(S))
@inline LinearAlgebra.eigvals(S::SymmetricTensor{2,dim,T}) where{dim,T} = convert(Vec{dim,T}, Vec{dim}(eigvals(to_smatrix(ustrip(S)))))

"""
    eigvecs(::SymmetricTensor)

Compute the eigenvectors of a symmetric tensor.
"""
@inline LinearAlgebra.eigvecs(S::SymmetricTensor{4}) = eigvecs(eigen(S))
@inline LinearAlgebra.eigvecs(S::SymmetricTensor{2}) = eigvecs(to_smatrix(ustrip(S)))

struct Eigen{T, S, dim, M}
    values::Vec{dim, T}
    vectors::Tensor{2, dim, S, M}
end

struct FourthOrderEigen{dim,T,S,M}
    values::Vector{T}
    vectors::Vector{SymmetricTensor{2,dim,S,M}}
end

# destructure via iteration
function Base.iterate(E::Union{Eigen,FourthOrderEigen}, state::Int=1)
    return iterate((eigvals(E), eigvecs(E)), state)
end

"""
    eigen(A::SymmetricTensor{2})

Compute the eigenvalues and eigenvectors of a symmetric second order tensor
and return an `Eigen` object. The eigenvalues are stored in a `Vec`,
sorted in ascending order. The corresponding eigenvectors are stored
as the columns of a `Tensor`.

See [`eigvals`](@ref) and [`eigvecs`](@ref).

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> E = eigen(A);

julia> E.values
2-element Vec{2, Float64}:
 -0.1883547111127678
  1.345436766284664

julia> E.vectors
2×2 Tensor{2, 2, Float64, 4}:
 -0.701412  0.712756
  0.712756  0.701412
```
"""
LinearAlgebra.eigen(::SymmetricTensor{2})

"""
    eigvals(::Union{Eigen,FourthOrderEigen})

Extract eigenvalues from an `Eigen` or `FourthOrderEigen` object,
returned by [`eigen`](@ref).
"""
@inline LinearAlgebra.eigvals(E::Union{Eigen,FourthOrderEigen}) = E.values
"""
    eigvecs(::Union{Eigen,FourthOrderEigen})

Extract eigenvectors from an `Eigen` or `FourthOrderEigen` object,
returned by [`eigen`](@ref).
"""
@inline LinearAlgebra.eigvecs(E::Union{Eigen,FourthOrderEigen}) = E.vectors

"""
    eigen(A::SymmetricTensor{4})

Compute the eigenvalues and second order eigentensors of a symmetric fourth
order tensor and return an `FourthOrderEigen` object. The eigenvalues and
eigentensors are sorted in ascending order of the eigenvalues.

See also [`eigvals`](@ref) and [`eigvecs`](@ref).
"""
function LinearAlgebra.eigen(R::SymmetricTensor{4,dim,T′}) where {dim,T′}
    S = ustrip(R)
    T = eltype(S)
    E = eigen(Hermitian(tomandel(S)))
    values = E.values isa Vector{T′} ? E.values : T′[T′(v) for v in E.values]
    vectors = [frommandel(SymmetricTensor{2,dim,T}, view(E.vectors, :, i)) for i in 1:size(E.vectors, 2)]
    return FourthOrderEigen(values, vectors)
end

# Use specialized for 1d as this can be inlined
@inline function LinearAlgebra.eigen(S::SymmetricTensor{2, 1, T}) where {T}
    @inbounds Eigen(Vec{1, T}((S[1, 1],)), one(Tensor{2, 1, T}))
end

function LinearAlgebra.eigen(R::SymmetricTensor{2,dim,T}) where{dim,T}
    e = eigen(to_smatrix(ustrip(R)))
    return Tensors.Eigen(convert(Vec{dim,T}, Vec{dim}(e.values)), Tensor{2,dim}(e.vectors))
end