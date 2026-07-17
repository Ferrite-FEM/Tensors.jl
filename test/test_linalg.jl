@testset "norm, trace, det, inv, eig" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
    # norm
    for order in (1,2,4)
        t = rand(Tensor{order, dim, T})
        @test (@inferred norm(t))::T ≈ sqrt(sum(abs2, Array(t)[:]))
        @test norm((@inferred normalize(t))::Tensor{order, dim, T}) ≈ one(T)
        if order != 1
            t_sym = rand(SymmetricTensor{order, dim, T})
            @test (@inferred norm(t_sym))::T ≈ sqrt(sum(abs2, Array(t_sym)[:]))
            @test norm((@inferred normalize(t_sym))::SymmetricTensor{order, dim, T}) ≈ one(T)
        end
    end

    # trace, vol, dev, det, inv (only for second order tensors)
    t = rand(Tensor{2, dim, T})
    t_sym = rand(SymmetricTensor{2, dim, T})

    @test (@inferred tr(t))::T == sum([t[i,i] for i in 1:dim])
    @test (@inferred tr(t_sym))::T == sum([t_sym[i,i] for i in 1:dim])

    @test tr(t) ≈ mean(t)*3.0
    @test tr(t_sym) ≈ mean(t_sym)*3.0

    @test (@inferred vol(t))::Tensor{2, dim, T} ≈ mean(t) * Matrix(I, dim, dim)
    @test (@inferred vol(t_sym))::SymmetricTensor{2, dim, T} ≈ mean(t_sym) * Matrix(I, dim, dim)

    @test (@inferred dev(t))::Tensor{2, dim, T} ≈ Array(t) - 1/3*tr(t)* Matrix(I, dim, dim)
    @test (@inferred dev(t_sym))::SymmetricTensor{2, dim, T} ≈ Array(t_sym) - 1/3*tr(t_sym)* Matrix(I, dim, dim)

    @test (@inferred det(t))::T ≈ det(Array(t))
    @test (@inferred det(t_sym))::T ≈ det(Array(t_sym))

    @test (@inferred inv(t))::Tensor{2, dim, T} ≈ inv(Array(t))
    @test (@inferred inv(t_sym))::SymmetricTensor{2, dim, T} ≈ inv(Array(t_sym))

    # inv for fourth order tensors
    Random.seed!(1234)
    AA = rand(Tensor{4, dim, T})
    AA_sym = rand(SymmetricTensor{4, dim, T})
    @test AA ⊡ (@inferred inv(AA))::Tensor{4, dim, T} ≈ one(Tensor{4, dim, T})
    @test AA_sym ⊡ (@inferred inv(AA_sym))::SymmetricTensor{4, dim, T} ≈ one(SymmetricTensor{4, dim, T})

    E = @inferred eigen(t_sym)
    Λ, Φ = E
    Λa, Φa = eigen(Array(t_sym))

    @test Λ ≈ (@inferred eigvals(t_sym)) ≈ eigvals(E) ≈ Λa
    @test Φ ≈ (@inferred eigvecs(t_sym)) ≈ eigvecs(E)
    for i in 1:dim
        # scale with first element of eigenvector to account for possible directions
        @test Φ[:, i]*Φ[1, i] ≈ Φa[:, i]*Φa[1, i]
    end

    # test eigenfactorizations for a diagonal tensor
    v = rand(T, dim)
    d_sym = diagm(SymmetricTensor{2, dim, T}, v)
    E = @inferred eigen(d_sym)
    Λ, Φ = E
    Λa, Φa = eigen(Symmetric(Array(d_sym)))

    @test Λ ≈ (@inferred eigvals(d_sym)) ≈ eigvals(E) ≈ Λa
    @test Φ ≈ (@inferred eigvecs(d_sym)) ≈ eigvecs(E)

    # sqrt
    Apd = tdot(t_sym)
    @test sqrt(Apd) ⋅ sqrt(Apd) ≈ Apd
end
end # of testset

@testset "eigen(::FourthOrderTensor)" begin
for T in (Float32, Float64), dim in (1, 2, 3)
    Random.seed!(123)
    # construct positive definite Voigt-tensor
    n = dim*dim - div((dim-1)*dim, 2)
    A = rand(T, n, n); A = A'A + I
    Aval, Avec = eigen(Hermitian(A))
    perm = sortperm(Aval)
    Aval = Aval[perm]
    Avec = [Avec[:, i] for i in perm]

    S = frommandel(SymmetricTensor{4,dim,T}, A)

    E = eigen(S)
    @test eigvals(E) ≈ Aval
    S′ = zero(S)
    for i in 1:n
        m = tomandel(eigvecs(E)[i])
        @test m / m[1] ≈ Avec[i] / Avec[i][1]
        @test S ⊡ E.vectors[i] ≈ E.values[i] * E.vectors[i]
        @test norm(E.vectors[i]) ≈ 1
        for j in 1:n
            if i == j
                @test E.vectors[i] ⊡ E.vectors[j] ≈ 1
            else
                @test E.vectors[i] ⊡ E.vectors[j] ≈ 0 atol=10eps(T)
            end
        end
        S′ += E.values[i] * (E.vectors[i] ⊗ E.vectors[i])
    end
    @test S ≈ S′
    a, b = E # iteration
    @test a == eigvals(E) == E.values
    @test b == eigvecs(E) == E.vectors
end
end

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
@testset "issues #100, #101: orders of eigenvectors" begin
    for (i, j) in ((1, 2), (2, 1))
        S = SymmetricTensor{2,2,Float64}((i, 0, j))
        Λ = diagm(SymmetricTensor{2,2}, eigvals(S))
        Φ = eigvecs(S)
        @test Φ ⋅ Λ ⋅ Φ' ≈ S
    end
    for (i, j, k) in ((1, 2, 3), (1, 3, 2), (2, 1, 3), (2, 3, 1), (3, 1, 2), (3, 2, 1))
        S = SymmetricTensor{2,3,Float64}((i, 0, 0, j, 0, k))
        Λ = diagm(SymmetricTensor{2,3}, eigvals(S))
        Φ = eigvecs(S)
        @test Φ ⋅ Λ ⋅ Φ' ≈ S
    end
end

@testset "issue #166: NaN in eigenvectors" begin
    M = [ 50.0  -10.0  0.0
         -10.0   40.0  0.0
           0.0    0.0  0.0]
    ME = eigen(Hermitian(M))
    S = SymmetricTensor{2,3}(M)
    SE = eigen(S)
    @test eigvals(SE) ≈ eigvals(ME)
    @test eigvecs(SE) ≈ eigvecs(ME)
end

