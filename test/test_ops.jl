function _permutedims{dim}(S::FourthOrderTensor{dim}, idx::NTuple{4,Int})
    sort([idx...]) == [1,2,3,4] || throw(ArgumentError("Missing index."))
    neworder = sortperm([idx...])
    f = (i,j,k,l) -> S[[i,j,k,l][neworder]...]
    return Tensor{4,dim}(f)
end

@testsection "tensor ops" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
println("T = $T, dim = $dim")
AA = rand(Tensor{4, dim, T})
BB = rand(Tensor{4, dim, T})
A = rand(Tensor{2, dim, T})
B = rand(Tensor{2, dim, T})
a = rand(Tensor{1, dim, T})
b = rand(Tensor{1, dim, T})

AA_sym = rand(SymmetricTensor{4, dim, T})
BB_sym = rand(SymmetricTensor{4, dim, T})
A_sym = rand(SymmetricTensor{2, dim, T})
B_sym = rand(SymmetricTensor{2, dim, T})

i,j,k,l = rand(1:dim,4)

@testsection "double contraction" begin
    # 4 - 4
    # Value tests
    @test vec(dcontract(AA, BB)) ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec(dcontract(AA_sym, BB)) ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec(dcontract(AA, BB_sym)) ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test vec(dcontract(AA_sym, BB_sym)) ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym)) ≈ dcontract(AA_sym, BB_sym)
    # Type tests
    @test isa(dcontract(AA, BB), Tensor{4, dim, T})
    @test isa(dcontract(AA_sym, BB), Tensor{4, dim, T})
    @test isa(dcontract(AA, BB_sym), Tensor{4, dim, T})
    @test isa(dcontract(AA_sym, BB_sym), SymmetricTensor{4, dim, T})

    # 2 - 4
    # Value tests
    @test dcontract(AA, A) ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test dcontract(AA_sym, A) ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test dcontract(AA, A_sym) ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(AA_sym, A_sym) ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(A, AA) ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test dcontract(A_sym, AA) ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(A, AA_sym) ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test dcontract(A_sym, AA_sym) ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym)) ≈ dcontract(AA_sym, A_sym)
    # Type tests
    @test isa(dcontract(AA, A), Tensor{2, dim, T})
    @test isa(dcontract(AA_sym, A), SymmetricTensor{2, dim, T})
    @test isa(dcontract(AA, A_sym), Tensor{2, dim, T})
    @test isa(dcontract(AA_sym, A_sym), SymmetricTensor{2, dim, T})
    @test isa(dcontract(A, AA), Tensor{2, dim, T})
    @test isa(dcontract(A_sym, AA), Tensor{2, dim, T})
    @test isa(dcontract(A, AA_sym), SymmetricTensor{2, dim, T})
    @test isa(dcontract(A_sym, AA_sym), SymmetricTensor{2, dim, T})

    # 2 - 2
    # Value tests
    @test dcontract(A, B) ≈ sum(vec(A) .* vec(B))
    @test dcontract(A_sym, B) ≈ sum(vec(A_sym) .* vec(B))
    @test dcontract(A, B_sym) ≈ sum(vec(A) .* vec(B_sym))
    @test dcontract(A_sym, B_sym) ≈ sum(vec(A_sym) .* vec(B_sym))
    # Type tests
    @test isa(dcontract(A, B), T)
    @test isa(dcontract(A_sym, B), T)
    @test isa(dcontract(A, B_sym), T)
    @test isa(dcontract(A_sym, B_sym), T)
end # of testsection

@testsection "outer product" begin
    # Value tests
    @test otimes(a, b) ≈ Array(a) * Array(b)'
    @test reshape(vec(otimes(A, B)), dim^2, dim^2) ≈ vec(A) * vec(B)'
    @test reshape(vec(otimes(A_sym, B)), dim^2, dim^2) ≈ vec(A_sym) * vec(B)'
    @test reshape(vec(otimes(A, B_sym)), dim^2, dim^2) ≈ vec(A) * vec(B_sym)'
    @test reshape(vec(otimes(A_sym, B_sym)), dim^2, dim^2) ≈ vec(A_sym) * vec(B_sym)'

    # Type tests
    @test isa(otimes(a, b), Tensor{2, dim, T})
    @test isa(otimes(A, B), Tensor{4, dim, T})
    @test isa(otimes(A_sym, B), Tensor{4, dim, T})
    @test isa(otimes(A, B_sym), Tensor{4, dim, T})
    @test isa(otimes(A_sym, B_sym), SymmetricTensor{4, dim, T})
end # of testsection

@testsection "dot products" begin
    # 1 - 2
    # Value tests
    @test dot(a, b) ≈ sum(Array(a) .* Array(b))
    @test dot(A, b) ≈ Array(A) * Array(b)
    @test dot(A_sym, b) ≈ Array(A_sym) * Array(b)
    @test dot(a, B) ≈ Array(B)' * Array(a)
    @test dot(a, B_sym) ≈ Array(B_sym)' * Array(a)

    # Type tests
    @test isa(dot(a, b), T)
    @test isa(dot(A, b), Vec{dim, T})
    @test isa(dot(A_sym, b), Vec{dim, T})
    @test isa(dot(b, A), Vec{dim, T})
    @test isa(dot(b, A_sym), Vec{dim, T})

    # 2 - 2
    # Value tests
    @test dot(A, B) ≈ Array(A) * Array(B)
    @test dot(A_sym, B) ≈ Array(A_sym) * Array(B)
    @test dot(A, B_sym) ≈ Array(A) * Array(B_sym)
    @test dot(A_sym, B_sym) ≈ Array(A_sym) * Array(B_sym)

    @test tdot(A) ≈ Array(A)' * Array(A)
    @test tdot(A_sym) ≈ Array(A_sym)' * Array(A_sym)
    @test dott(A) ≈ Array(A) * Array(A)'
    @test dott(A_sym) ≈ Array(A_sym) * Array(A_sym)'
    @test tdot(A) ≈ dott(transpose(A))
    @test tdot(transpose(A)) ≈ dott(A)
    @test tdot(A_sym) ≈ dott(transpose(A_sym))
    @test tdot(transpose(A_sym)) ≈ dott(A_sym)

    # Type tests
    @test isa(dot(A, B), Tensor{2, dim, T})
    @test isa(dot(A_sym, B), Tensor{2, dim, T})
    @test isa(dot(A, B_sym), Tensor{2, dim, T})
    @test isa(dot(A_sym, B_sym), Tensor{2, dim, T})

    @test isa(tdot(A), SymmetricTensor{2, dim, T})
    @test isa(tdot(A_sym), SymmetricTensor{2, dim, T})
    @test isa(dott(A), SymmetricTensor{2, dim, T})
    @test isa(dott(A_sym), SymmetricTensor{2, dim, T})
end # of testsection

@testsection "symmetric/skew-symmetric" begin
    if dim == 1 # non-symmetric tensors are symmetric
        @test issymmetric(A)
        @test issymmetric(AA)
    elseif dim != 1
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
    @test isa(symmetric(A), SymmetricTensor{2, dim, T})
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

    @test isa(minorsymmetric(AA), SymmetricTensor{4, dim, T})
    @test convert(typeof(AA_sym),convert(Tensor,minorsymmetric(AA))) ≈ minorsymmetric(AA)

    @test majorsymmetric(AA)[i,j,k,l] ≈ 0.5*(AA[i,j,k,l] + AA[k,l,i,j])
    @test majorsymmetric(AA)[i,j,k,l] ≈ majorsymmetric(AA)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA))
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ 0.5*(AA_sym[i,j,k,l] + AA_sym[k,l,i,j])
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ majorsymmetric(AA_sym)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA_sym))
    @test isa(majorsymmetric(AA), Tensor{4, dim, T})
    @test isa(majorsymmetric(AA_sym), Tensor{4, dim, T})

    @test skew(A) ≈ 0.5(A - A.')
    @test isa(skew(A), Tensor{2, dim, T})
    @test skew(A_sym) ≈ zero(A_sym)
    @test isa(skew(A_sym), Tensor{2, dim, T})

    # Identities
    @test A ≈ symmetric(A) + skew(A)
    @test skew(A) ≈ -skew(A).'
    @test trace(skew(A)) ≈ 0.0
    @test trace(symmetric(A)) ≈ trace(A)
end # of testsection

@testsection "transpose" begin
    @test transpose(a) ≈ a' ≈ a
    @test isa(transpose(a), Vec{dim, T})
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

    @test minortranspose(AA) ≈ _permutedims(AA,(2,1,4,3))
    @test minortranspose(AA_sym) ≈ _permutedims(AA_sym,(2,1,4,3))
    @test majortranspose(AA) ≈ _permutedims(AA,(3,4,1,2))
    @test majortranspose(AA_sym) ≈ _permutedims(AA_sym,(3,4,1,2))
end # of testsection

@testsection "cross product" begin
    @test a × a ≈ Vec{3, T}((0.0,0.0,0.0))
    @test a × b ≈ -b × a
    if dim == 2
        ad = Vec{2, T}((1.0,0.0))
        ad2 = Vec{2, T}((0.0,1.0))
        @test ad × ad2 ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
    if dim == 3
        ad = Vec{3, T}((1.0,0.0,0.0))
        ad2 = Vec{3, T}((0.0,1.0,0.0))
        @test ad × ad2 ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
end # of testsection

@testsection "special" begin
    AAT = Tensor{4, dim, T}((i,j,k,l) -> AA_sym[i,l,k,j])
    @test AAT ⊡ (b ⊗ a) ≈ dotdot(a, AA_sym, b)
end # of testsection

@testsection "rotation" begin
    x = eᵢ(Vec{3}, 1)
    y = eᵢ(Vec{3}, 2)
    z = eᵢ(Vec{3}, 3)

    @test rotate(z, z, rand()) ≈ z
    @test rotate(2*z, y, π/2) ≈ 2*x
    @test rotate(3*z, y, π) ≈ -3*z
    @test rotate(x+y+z, z, π/4) ≈ Vec{3}((0.0,√2,1.0))

    a = rand(Vec{3})
    b = rand(Vec{3})
    @test rotate(a, b, 0) ≈ a
    @test rotate(a, b, π) ≈ rotate(a, b, -π)
    @test rotate(a, b, π/2) ≈ rotate(a, -b, -π/2)
end

@testsection "tovoigt/fromvoigt" begin
    @test tovoigt(AA) * tovoigt(A) ≈ tovoigt(AA ⊡ A)
    @test tovoigt(AA_sym) * tovoigt(A_sym, offdiagscale=2) ≈ tovoigt(AA_sym ⊡ A_sym)

    @test fromvoigt(Tensor{2,dim}, tovoigt(A)) ≈ A
    @test fromvoigt(Tensor{4,dim}, tovoigt(AA)) ≈ AA
    @test fromvoigt(SymmetricTensor{2,dim}, tovoigt(A_sym, offdiagscale=2), offdiagscale=2) ≈ A_sym
    @test fromvoigt(SymmetricTensor{4,dim}, tovoigt(AA_sym, offdiagscale=2), offdiagscale=2) ≈ AA_sym

    @test tomandel(AA_sym) * tomandel(A_sym) ≈ tomandel(AA_sym ⊡ A_sym)
    @test frommandel(SymmetricTensor{2,dim}, tomandel(A_sym)) ≈ A_sym
    @test frommandel(SymmetricTensor{4,dim}, tomandel(AA_sym)) ≈ AA_sym
end
end # of testsection
end # of testsection