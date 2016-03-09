# Test dcontract
using ContMechTensors
using Base.Test

# Dcontract
for dim in (1,2,3)
    println(dim)
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

    # 4 - 4
    @test vec(dcontract(AA, BB)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
    @test vec(dcontract(AA_sym, BB_sym)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym)) ≈ dcontract(AA_sym, BB_sym)
    @test vec(dcontract(AA_sym, BB)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
    @test vec(dcontract(AA, BB_sym)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))

    # 2 - 4
    @test dcontract(AA, A) ≈ reshape(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(A, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(AA_sym, A_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(A_sym, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(A, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(A_sym, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
    @test dcontract(A, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym)) ≈ dcontract(AA_sym, A_sym)


    # 2 - 2
    @test dcontract(A, B) ≈ sum(vec(A) .* vec(B))
    @test dcontract(A_sym, B_sym) ≈ sum(vec(A_sym) .* vec(B_sym))
    @test dcontract(A, B_sym) ≈ sum(vec(A) .* vec(B_sym))
    @test dcontract(A_sym, B) ≈ sum(vec(A_sym) .* vec(B))
end


