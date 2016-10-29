@testset "tensor operations" begin; for dim in (1,2,3)
AA = rand(Tensor{4, dim})
BB = rand(Tensor{4, dim})
A = rand(Tensor{2, dim})
B = rand(Tensor{2, dim})
a = rand(Tensor{1, dim})
b = rand(Tensor{1, dim})

AA_sym = rand(SymmetricTensor{4, dim})
BB_sym = rand(SymmetricTensor{4, dim})
A_sym = rand(SymmetricTensor{2, dim})
B_sym = rand(SymmetricTensor{2, dim})

i,j,k,l = rand(1:dim,4)

@testset "double contraction" begin
    # 4 - 4
    # Value tests
    @test vec(dcontract(AA, BB)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
    @test vec(dcontract(AA_sym, BB)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
    @test vec(dcontract(AA, BB_sym)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))
    @test vec(dcontract(AA_sym, BB_sym)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym)) ≈ dcontract(AA_sym, BB_sym)
    # Type tests
    @test isa(dcontract(AA, BB), Tensor{4,dim})
    @test isa(dcontract(AA_sym, BB), Tensor{4,dim})
    @test isa(dcontract(AA, BB_sym), Tensor{4,dim})
    @test isa(dcontract(AA_sym, BB_sym), SymmetricTensor{4,dim})

    # 2 - 4
    # Value tests
    @test dcontract(AA, A) ≈ reshape(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(AA_sym, A) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(AA, A_sym) ≈ reshape(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(AA_sym, A_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(A, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(A_sym, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(A, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(A_sym, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym)) ≈ dcontract(AA_sym, A_sym)
    # Type tests
    @test isa(dcontract(AA, A), Tensor{2,dim})
    @test isa(dcontract(AA_sym, A), Tensor{2,dim})
    @test isa(dcontract(AA, A_sym), Tensor{2,dim})
    @test isa(dcontract(AA_sym, A_sym), SymmetricTensor{2,dim})
    @test isa(dcontract(A, AA), Tensor{2,dim})
    @test isa(dcontract(A_sym, AA), Tensor{2,dim})
    @test isa(dcontract(A, AA_sym), Tensor{2,dim})
    @test isa(dcontract(A_sym, AA_sym), SymmetricTensor{2,dim})

    # 2 - 2
    # Value tests
    @test dcontract(A, B) ≈ sum(vec(A) .* vec(B))
    @test dcontract(A_sym, B) ≈ sum(vec(A_sym) .* vec(B))
    @test dcontract(A, B_sym) ≈ sum(vec(A) .* vec(B_sym))
    @test dcontract(A_sym, B_sym) ≈ sum(vec(A_sym) .* vec(B_sym))
    # Type tests
    @test isa(dcontract(A, B), Real)
    @test isa(dcontract(A_sym, B), Real)
    @test isa(dcontract(A, B_sym), Real)
    @test isa(dcontract(A_sym, B_sym), Real)
end # of testset

@testset "outer product" begin
    # Value tests
    @test otimes(a, b) ≈ Array(a) * Array(b)'
    @test reshape(vec(otimes(A, B)), dim^2, dim^2) ≈ vec(A) * vec(B)'
    @test reshape(vec(otimes(A_sym, B)), dim^2, dim^2) ≈ vec(A_sym) * vec(B)'
    @test reshape(vec(otimes(A, B_sym)), dim^2, dim^2) ≈ vec(A) * vec(B_sym)'
    @test reshape(vec(otimes(A_sym, B_sym)), dim^2, dim^2) ≈ vec(A_sym) * vec(B_sym)'

    # Type tests
    @test isa(otimes(a, b), Tensor{2,dim})
    @test isa(otimes(A, B), Tensor{4,dim})
    @test isa(otimes(A_sym, B), Tensor{4,dim})
    @test isa(otimes(A, B_sym), Tensor{4,dim})
    @test isa(otimes(A_sym, B_sym), SymmetricTensor{4,dim})
end # of testset

@testset "dot products" begin
    # 1 - 2
    # Value tests
    @test dot(a, b) ≈ sum(Array(a) .* Array(b))
    @test dot(A, b) ≈ Array(A) * Array(b)
    @test dot(A_sym, b) ≈ Array(A_sym) * Array(b)
    @test dot(a, B) ≈ Array(B)' * Array(a)
    @test dot(a, B_sym) ≈ Array(B_sym)' * Array(a)
    # Type tests
    @test isa(dot(a, b), Real)
    @test isa(dot(A, b), Tensor{1,dim})
    @test isa(dot(A_sym, b), Tensor{1,dim})
    @test isa(dot(b, A), Tensor{1,dim})
    @test isa(dot(b, A_sym), Tensor{1,dim})

    # 2 - 2
    # Value tests
    @test dot(A, B) ≈ Array(A) * Array(B)
    @test dot(A_sym, B) ≈ Array(A_sym) * Array(B)
    @test dot(A, B_sym) ≈ Array(A) * Array(B_sym)
    @test dot(A_sym, B_sym) ≈ Array(A_sym) * Array(B_sym)

    @test tdot(A, B) ≈ Array(A)' * Array(B)
    @test tdot(A_sym, B) ≈ Array(A_sym)' * Array(B)
    @test tdot(A, B_sym) ≈ Array(A)' * Array(B_sym)
    @test tdot(A_sym, B_sym) ≈ Array(A_sym)' * Array(B_sym)
    @test tdot(A) ≈ Array(A)' * Array(A)
    @test tdot(A_sym) ≈ Array(A_sym)' * Array(A_sym)

    # Type tests
    @test isa(dot(A, B), Tensor{2,dim})
    @test isa(dot(A_sym, B), Tensor{2,dim})
    @test isa(dot(A, B_sym), Tensor{2,dim})
    @test isa(dot(A_sym, B_sym), Tensor{2,dim})

    @test isa(tdot(A, B), Tensor{2,dim})
    @test isa(tdot(A_sym, B), Tensor{2,dim})
    @test isa(tdot(A, B_sym), Tensor{2,dim})
    @test isa(tdot(A_sym, B_sym), Tensor{2,dim})
    @test isa(tdot(A), SymmetricTensor{2,dim})
    @test isa(tdot(A_sym), SymmetricTensor{2,dim})
end # of testset

@testset "determinant" begin
    @test det(A) ≈ det(reshape(vec(A), (dim,dim)))
    @test det(A_sym) ≈ det(reshape(vec(A_sym), (dim,dim)))
end # of testset

@testset "symmetric/skew-symmetric" begin
    if dim != 1
        @test !issymmetric(A)
        @test !issymmetric(AA)
        @test !ismajorsymmetric(AA)
        @test !isminorsymmetric(AA)
        @test_throws InexactError convert(typeof(A_sym),A)
        @test_throws InexactError convert(typeof(AA_sym),AA)
    end
    @test issymmetric(A_sym)
    @test issymmetric(AA_sym)
    @test isminorsymmetric(AA_sym)
    @test issymmetric(symmetric(A))
    @test issymmetric(A + A.')

    @test symmetric(A) ≈ 0.5(A + A.')
    @test symmetric(A_sym) ≈ A_sym
    @test typeof(symmetric(A)) <: SymmetricTensor{2,dim}
    @test convert(typeof(A_sym),convert(Tensor,symmetric(A))) ≈ symmetric(A)

    @test symmetric(AA) ≈ minorsymmetric(AA)
    @test minorsymmetric(AA)[i,j,k,l] ≈ minorsymmetric(AA)[j,i,l,k]
    @test issymmetric(convert(Tensor,minorsymmetric(AA)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA)))
    @test symmetric(AA_sym) ≈ minorsymmetric(AA_sym)
    @test minorsymmetric(AA_sym)[i,j,k,l] ≈ minorsymmetric(AA_sym)[j,i,l,k]
    @test minorsymmetric(AA_sym) ≈ AA_sym
    @test issymmetric(convert(Tensor,minorsymmetric(AA_sym)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA_sym)))

    @test typeof(minorsymmetric(AA)) <: SymmetricTensor{4,dim}
    @test convert(typeof(AA_sym),convert(Tensor,minorsymmetric(AA))) ≈ minorsymmetric(AA)

    @test majorsymmetric(AA)[i,j,k,l] ≈ 0.5*(AA[i,j,k,l] + AA[k,l,i,j])
    @test majorsymmetric(AA)[i,j,k,l] ≈ majorsymmetric(AA)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA))
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ 0.5*(AA_sym[i,j,k,l] + AA_sym[k,l,i,j])
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ majorsymmetric(AA_sym)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA_sym))
    @test typeof(majorsymmetric(AA)) <: Tensor{4,dim}
    @test typeof(majorsymmetric(AA_sym)) <: Tensor{4,dim}

    @test skew(A) ≈ 0.5(A - A.')
    @test skew(A_sym) ≈ zero(A_sym)
    @test typeof(skew(A_sym)) <: Tensor{2,dim}

    # Identities
    @test A ≈ symmetric(A) + skew(A)
    @test skew(A) ≈ -skew(A).'
    @test trace(skew(A)) ≈ 0.0
    @test trace(symmetric(A)) ≈ trace(A)
end # of testset

@testset "transpose" begin
    @test transpose(a) ≈ a' ≈ a
    @test isa(transpose(a), Vec{dim})
    @test transpose(A) ≈ Array(A).'
    @test transpose(transpose(A)) ≈ A
    @test transpose(A_sym) ≈ Array(A_sym).'
    @test transpose(A_sym) ≈ A_sym
    @test transpose(transpose(A_sym)) ≈ A_sym

    @test transpose(AA) ≈ minortranspose(AA)
    @test AA[i,j,k,l] ≈ minortranspose(AA)[j,i,l,k]
    @test AA_sym[i,j,k,l] ≈ minortranspose(AA_sym)[j,i,l,k]
    @test minortranspose(AA_sym) ≈ AA_sym
    @test minortranspose(minortranspose(AA)) ≈ AA
    @test minortranspose(minortranspose(AA_sym)) ≈ AA_sym

    @test majortranspose(majortranspose(AA)) ≈ AA
    @test majortranspose(majortranspose(AA_sym)) ≈ AA_sym
    @test AA[i,j,k,l] ≈ majortranspose(AA)[k,l,i,j]
    @test AA_sym[i,j,k,l] ≈ majortranspose(AA_sym)[k,l,i,j]
    @test typeof(majortranspose(AA_sym)) <: Tensor{4,dim}
end # of testset

@testset "permute_index" begin
    @test permute_index(AA,(1,2,3,4)) ≈ AA
    @test permute_index(AA_sym,(1,2,3,4)) ≈ AA_sym
    @test permute_index(AA,(2,1,4,3)) ≈ minortranspose(AA)
    @test permute_index(AA_sym,(2,1,4,3)) ≈ minortranspose(AA_sym)
    @test permute_index(AA,(3,4,1,2)) ≈ majortranspose(AA)
    @test permute_index(AA_sym,(3,4,1,2)) ≈ majortranspose(AA_sym)
    @test permute_index(permute_index(AA,(1,4,2,3)),(1,3,4,2)) ≈ AA
    @test permute_index(permute_index(AA_sym,(1,4,2,3)),(1,3,4,2)) ≈ AA_sym
    @test typeof(permute_index(AA,(1,4,3,2))) <: Tensor{4,dim}
    @test typeof(permute_index(AA_sym,(1,4,3,2))) <: Tensor{4,dim}
    @test_throws ArgumentError permute_index(AA,(1,1,2,3))
end # of testset

@testset "cross product" begin
    @test a × a ≈ Vec{3}((0.0,0.0,0.0))
    @test a × b ≈ -b × a
    if dim == 2
        ad = Vec{2}((1.0,0.0))
        ad2 = Vec{2}((0.0,1.0))
        @test ad × ad2 ≈ Vec{3}((0.0, 0.0, 1.0))
    end
    if dim == 3
        ad = Vec{3}((1.0,0.0,0.0))
        ad2 = Vec{3}((0.0,1.0,0.0))
        @test ad × ad2 ≈ Vec{3}((0.0, 0.0, 1.0))
    end
end # of testset

@testset "special" begin
    AAT = Tensor{4, dim}((i,j,k,l) -> AA_sym[i,l,k,j])
    @test AAT ⊡ (b ⊗ a) ≈ dotdot(a, AA_sym, b)
end # of testset
end
end # of testset
