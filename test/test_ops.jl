@testset "tensor ops" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
AA = rand(Tensor{4, dim, T})
BB = rand(Tensor{4, dim, T})
A3 = rand(Tensor{3, dim, T})
B3 = rand(Tensor{3, dim, T})
A = rand(Tensor{2, dim, T})
B = rand(Tensor{2, dim, T})
a = rand(Tensor{1, dim, T})
b = rand(Tensor{1, dim, T})

AA_sym = rand(SymmetricTensor{4, dim, T})
BB_sym = rand(SymmetricTensor{4, dim, T})
A_sym = rand(SymmetricTensor{2, dim, T})
B_sym = rand(SymmetricTensor{2, dim, T})
symA = symmetric(A)
symB = symmetric(B)

i,j,k,l = rand(1:dim,4)

@testset "double contraction" begin
    # 4 - 4
    @test vec((@inferred dcontract(AA, BB))::Tensor{4, dim, T})                  ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA, BB_sym))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB_sym))::SymmetricTensor{4, dim, T}) ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym))            ≈ dcontract(AA_sym, BB_sym)

    # 3 - 4
    @test dcontract(AA, A3) ≈ Tensor{3,dim}((i,j,k) -> sum(AA[i,j,m,n]*A3[m,n,k] for m in 1:dim, n in 1:dim))
    @test dcontract(A3, AA) ≈ Tensor{3,dim}((i,j,k) -> sum(A3[i,m,n]*AA[m,n,j,k] for m in 1:dim, n in 1:dim))

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

    # 2 - 3
    @test dcontract(A, A3) ≈ Tensor{1,dim}((i) -> sum(A[m,n]*A3[m,n,i] for m in 1:dim, n in 1:dim))
    @test dcontract(A3, A) ≈ Tensor{1,dim}((i) -> sum(A3[i,m,n]*A[m,n] for m in 1:dim, n in 1:dim))

    # 2 - 2
    @test (@inferred dcontract(A, B))::T         ≈ sum(vec(A) .* vec(B))
    @test (@inferred dcontract(A_sym, B))::T     ≈ sum(vec(A_sym) .* vec(B))
    @test (@inferred dcontract(A, B_sym))::T     ≈ sum(vec(A) .* vec(B_sym))
    @test (@inferred dcontract(A_sym, B_sym))::T ≈ sum(vec(A_sym) .* vec(B_sym))
end # of testset

@testset "dot products" begin
    # 1 - 1
    @test (@inferred dot(a, b))::T               ≈ sum(Array(a) .* Array(b))
    # 1 - 2
    @test (@inferred dot(A, b))::Vec{dim, T}     ≈ Array(A) * Array(b)
    @test (@inferred dot(A_sym, b))::Vec{dim, T} ≈ Array(A_sym) * Array(b)
    @test (@inferred dot(a, B))::Vec{dim, T}     ≈ Array(B)' * Array(a)
    @test (@inferred dot(a, B_sym))::Vec{dim, T} ≈ Array(B_sym)' * Array(a)
    # 2 - 2
    # binary
    @test (@inferred dot(A, B))::Tensor{2, dim, T}         ≈ Array(A) * Array(B)
    @test (@inferred dot(A_sym, B))::Tensor{2, dim, T}     ≈ Array(A_sym) * Array(B)
    @test (@inferred dot(A, B_sym))::Tensor{2, dim, T}     ≈ Array(A) * Array(B_sym)
    @test (@inferred dot(A_sym, B_sym))::Tensor{2, dim, T} ≈ Array(A_sym) * Array(B_sym)
    # unary
    @test (@inferred dot(A_sym))::SymmetricTensor{2, dim, T}  ≈ dot(A_sym, A_sym)
    @test (@inferred tdot(A))::SymmetricTensor{2, dim, T}     ≈ dot(transpose(A), A)
    @test (@inferred tdot(A_sym))::SymmetricTensor{2, dim, T} ≈ dot(transpose(A_sym), A_sym)
    @test (@inferred dott(A))::SymmetricTensor{2, dim, T}     ≈ dot(A, transpose(A))
    @test (@inferred dott(A_sym))::SymmetricTensor{2, dim, T} ≈ dot(A_sym, transpose(A_sym))
    # 2 - 4
    @test (@inferred dot(AA, B))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA), (dim^3, dim))) * collect(reshape(vec(B), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B, AA))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B), (dim, dim))) * collect(reshape(vec(AA), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA_sym, B))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^3, dim))) * collect(reshape(vec(B), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B, AA_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B), (dim, dim))) * collect(reshape(vec(AA_sym), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA, B_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA), (dim^3, dim))) * collect(reshape(vec(B_sym), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B_sym, AA))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B_sym), (dim, dim))) * collect(reshape(vec(AA), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA_sym, B_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^3, dim))) * collect(reshape(vec(B_sym), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B_sym, AA_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B_sym), (dim, dim))) * collect(reshape(vec(AA_sym), (dim, dim^3))), (dim, dim, dim, dim))
end # of testset

@testset "cross product" begin
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
    if T == Float64 # mixed eltype
        @test rand(Vec{dim,Float64}) × rand(Vec{dim,Float32}) isa Vec{3,Float64}
    end
end # of testset

@testset "special" begin
    AAT = Tensor{4, dim, T}((i,j,k,l) -> AA_sym[i,l,k,j])
    @test AAT ⊡ (b ⊗ a) ≈ (@inferred dotdot(a, AA_sym, b))::Tensor{2, dim, T}
end # of testset
end
end # of testset
