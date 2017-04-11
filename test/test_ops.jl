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
    @test vec((@inferred dcontract(AA, BB))::Tensor{4, dim, T})                  ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA, BB_sym))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB_sym))::SymmetricTensor{4, dim, T}) ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym))            ≈ dcontract(AA_sym, BB_sym)

    # 2 - 4
    @test (@inferred dcontract(AA, A))::Tensor{2, dim, T}                  ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA_sym, A))::SymmetricTensor{2, dim, T}     ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA, A_sym))::Tensor{2, dim, T}              ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA_sym, A_sym))::SymmetricTensor{2, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A, AA))::Tensor{2, dim, T}                  ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A_sym, AA))::Tensor{2, dim, T}              ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A, AA_sym))::SymmetricTensor{2, dim, T}     ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A_sym, AA_sym))::SymmetricTensor{2, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym))       ≈ dcontract(AA_sym, A_sym)

    # 2 - 2
    @test (@inferred dcontract(A, B))::T         ≈ sum(vec(A) .* vec(B))
    @test (@inferred dcontract(A_sym, B))::T     ≈ sum(vec(A_sym) .* vec(B))
    @test (@inferred dcontract(A, B_sym))::T     ≈ sum(vec(A) .* vec(B_sym))
    @test (@inferred dcontract(A_sym, B_sym))::T ≈ sum(vec(A_sym) .* vec(B_sym))
end # of testsection

@testsection "outer product" begin
    @test             (@inferred otimes(a, b))::Tensor{2, dim, T}                                  ≈ Array(a) * Array(b)'
    @test reshape(vec((@inferred otimes(A, B))::Tensor{4, dim, T}), dim^2, dim^2)                  ≈ vec(A) * vec(B)'
    @test reshape(vec((@inferred otimes(A_sym, B))::Tensor{4, dim, T}), dim^2, dim^2)              ≈ vec(A_sym) * vec(B)'
    @test reshape(vec((@inferred otimes(A, B_sym))::Tensor{4, dim, T}), dim^2, dim^2)              ≈ vec(A) * vec(B_sym)'
    @test reshape(vec((@inferred otimes(A_sym, B_sym))::SymmetricTensor{4, dim, T}), dim^2, dim^2) ≈ vec(A_sym) * vec(B_sym)'
end # of testsection

@testsection "dot products" begin
    # 1 - 2
    @test (@inferred dot(a, b))::T               ≈ sum(Array(a) .* Array(b))
    @test (@inferred dot(A, b))::Vec{dim, T}     ≈ Array(A) * Array(b)
    @test (@inferred dot(A_sym, b))::Vec{dim, T} ≈ Array(A_sym) * Array(b)
    @test (@inferred dot(a, B))::Vec{dim, T}     ≈ Array(B)' * Array(a)
    @test (@inferred dot(a, B_sym))::Vec{dim, T} ≈ Array(B_sym)' * Array(a)

    # 2 - 2
    @test (@inferred dot(A, B))::Tensor{2, dim, T}         ≈ Array(A) * Array(B)
    @test (@inferred dot(A_sym, B))::Tensor{2, dim, T}     ≈ Array(A_sym) * Array(B)
    @test (@inferred dot(A, B_sym))::Tensor{2, dim, T}     ≈ Array(A) * Array(B_sym)
    @test (@inferred dot(A_sym, B_sym))::Tensor{2, dim, T} ≈ Array(A_sym) * Array(B_sym)

    @test (@inferred tdot(A))::SymmetricTensor{2, dim, T}     ≈ Array(A)' * Array(A)
    @test (@inferred tdot(A_sym))::SymmetricTensor{2, dim, T} ≈ Array(A_sym)' * Array(A_sym)
    @test (@inferred dott(A))::SymmetricTensor{2, dim, T}     ≈ Array(A) * Array(A)'
    @test (@inferred dott(A_sym))::SymmetricTensor{2, dim, T} ≈ Array(A_sym) * Array(A_sym)'

    @test tdot(A) ≈ dott(transpose(A))
    @test tdot(transpose(A)) ≈ dott(A)
    @test tdot(A_sym) ≈ dott(transpose(A_sym))
    @test tdot(transpose(A_sym)) ≈ dott(A_sym)
end # of testsection

@testsection "symmetric/skew-symmetric" begin
    if dim == 1 # non-symmetric tensors are symmetric
        @test (@inferred issymmetric(A))
        @test (@inferred issymmetric(AA))
    elseif dim != 1
        @test !(@inferred issymmetric(A))
        @test !(@inferred issymmetric(AA))
        @test !(@inferred ismajorsymmetric(AA))
        @test !(@inferred isminorsymmetric(AA))
        @test_throws InexactError convert(typeof(A_sym),A)
        @test_throws InexactError convert(typeof(AA_sym),AA)
    end
    @test (@inferred issymmetric(A_sym))
    @test (@inferred issymmetric(AA_sym))
    @test (@inferred isminorsymmetric(AA_sym))
    @test (@inferred issymmetric(symmetric(A)))
    @test (@inferred issymmetric(A + A.'))

    @test (@inferred symmetric(A))::SymmetricTensor{2, dim, T}     ≈ 0.5(A + A.')
    @test (@inferred symmetric(A_sym))::SymmetricTensor{2, dim, T} ≈ A_sym
    @test convert(typeof(A_sym),convert(Tensor,symmetric(A))) ≈ symmetric(A)

    @test (@inferred symmetric(AA))::SymmetricTensor{4, dim, T} ≈ (@inferred minorsymmetric(AA))::SymmetricTensor{4, dim, T}
    @test minorsymmetric(AA)[i,j,k,l] ≈ minorsymmetric(AA)[j,i,l,k]
    @test issymmetric(convert(Tensor,minorsymmetric(AA)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA)))
    @test (@inferred symmetric(AA_sym))::SymmetricTensor{4, dim, T} ≈ (@inferred minorsymmetric(AA_sym))::SymmetricTensor{4, dim, T}
    @test minorsymmetric(AA_sym)[i,j,k,l] ≈ minorsymmetric(AA_sym)[j,i,l,k]
    @test minorsymmetric(AA_sym) ≈ AA_sym
    @test issymmetric(convert(Tensor,minorsymmetric(AA_sym)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA_sym)))

    @test convert(typeof(AA_sym),convert(Tensor,minorsymmetric(AA))) ≈ minorsymmetric(AA)

    @test ((@inferred majorsymmetric(AA))::Tensor{4, dim, T})[i,j,k,l] ≈ 0.5*(AA[i,j,k,l] + AA[k,l,i,j])
    @test majorsymmetric(AA)[i,j,k,l] ≈ majorsymmetric(AA)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA))
    @test ((@inferred majorsymmetric(AA_sym))::Tensor{4, dim, T})[i,j,k,l] ≈ 0.5*(AA_sym[i,j,k,l] + AA_sym[k,l,i,j])
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ majorsymmetric(AA_sym)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA_sym))

    @test (@inferred skew(A))::Tensor{2, dim, T} ≈ 0.5(A - A.')
    @test (@inferred skew(A_sym))::Tensor{2, dim, T} ≈ zero(A_sym)

    # Identities
    @test A ≈ symmetric(A) + skew(A)
    @test skew(A) ≈ -skew(A).'
    @test trace(skew(A)) ≈ 0.0
    @test trace(symmetric(A)) ≈ trace(A)
end # of testsection

@testsection "transpose" begin
    @test (@inferred transpose(a))::Vec{dim, T} ≈ a' ≈ a
    @test (@inferred transpose(A))::Tensor{2, dim, T} ≈ Array(A).'
    @test transpose(transpose(A)) ≈ A
    @test (@inferred transpose(A_sym))::SymmetricTensor{2, dim, T} ≈ A_sym ≈ Array(A_sym).'
    @test transpose(transpose(A_sym)) ≈ A_sym

    @test (@inferred transpose(AA))::Tensor{4, dim, T} ≈ (@inferred minortranspose(AA))::Tensor{4, dim, T}
    @test AA[i,j,k,l] ≈ minortranspose(AA)[j,i,l,k]
    @test AA_sym[i,j,k,l] ≈ minortranspose(AA_sym)[j,i,l,k]
    @test (@inferred transpose(AA_sym))::SymmetricTensor{4, dim, T} ≈ (@inferred minortranspose(AA_sym))::SymmetricTensor{4, dim, T} ≈ AA_sym
    @test minortranspose(minortranspose(AA)) ≈ AA
    @test minortranspose(minortranspose(AA_sym)) ≈ AA_sym

    @test majortranspose((@inferred majortranspose(AA))::Tensor{4, dim, T}) ≈ AA
    @test majortranspose((@inferred majortranspose(AA_sym))::Tensor{4, dim, T}) ≈ AA_sym
    @test AA[i,j,k,l] ≈ majortranspose(AA)[k,l,i,j]
    @test AA_sym[i,j,k,l] ≈ majortranspose(AA_sym)[k,l,i,j]

    @test minortranspose(AA) ≈ _permutedims(AA,(2,1,4,3))
    @test minortranspose(AA_sym) ≈ _permutedims(AA_sym,(2,1,4,3))
    @test majortranspose(AA) ≈ _permutedims(AA,(3,4,1,2))
    @test majortranspose(AA_sym) ≈ _permutedims(AA_sym,(3,4,1,2))
end # of testsection

@testsection "cross product" begin
    @test (@inferred a × a)::Vec{3, T} ≈ Vec{3, T}((0.0,0.0,0.0))
    @test a × b ≈ -b × a
    if dim == 2
        ad = Vec{2, T}((1.0,0.0))
        ad2 = Vec{2, T}((0.0,1.0))
        @test (@inferred ad × ad2)::Vec{3, T} ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
    if dim == 3
        ad = Vec{3, T}((1.0,0.0,0.0))
        ad2 = Vec{3, T}((0.0,1.0,0.0))
        @test (@inferred ad × ad2)::Vec{3, T} ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
end # of testsection

@testsection "special" begin
    AAT = Tensor{4, dim, T}((i,j,k,l) -> AA_sym[i,l,k,j])
    @test AAT ⊡ (b ⊗ a) ≈ (@inferred dotdot(a, AA_sym, b))::Tensor{2, dim, T}
end # of testsection

@testsection "rotation" begin
    x = eᵢ(Vec{3}, 1)
    y = eᵢ(Vec{3}, 2)
    z = eᵢ(Vec{3}, 3)

    @test (@inferred rotate(z, z, rand()))::Vec{dim, T} ≈ z
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
