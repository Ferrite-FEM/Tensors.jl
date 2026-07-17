# Package-hygiene guards (method ambiguities, allocations) and regression
# tests for review findings.

using Tensors, Test, LinearAlgebra
using Statistics: mean

@testset "hygiene" begin
    @testset "method ambiguities" begin
        # only the three pre-existing literal_pow/MixedTensor2 ambiguities are
        # tolerated; anything new is a regression
        ambiguities = Test.detect_ambiguities(Tensors; recursive = true)
        @test length(ambiguities) <= 3
        # known previously-ambiguous intersections dispatch fine
        @test norm(rand(Tensor{4, 3})) > 0
        @test norm(rand(Tensor{4, 3, Float32})) > 0
    end

    @testset "zero allocations on hot operations" begin
        a = rand(Vec{3}); A = rand(Tensor{2, 3}); As = rand(SymmetricTensor{2, 3})
        C = rand(Tensor{4, 3}); Cs = rand(SymmetricTensor{4, 3})
        m = rand(MixedTensor{2, Tuple{2, 3}})
        for f in (() -> a ⊗ a, () -> A ⋅ a, () -> A ⋅ A, () -> A ⊡ A, () -> As ⊡ As,
                  () -> C ⊡ A, () -> Cs ⊡ As, () -> C ⊡ C, () -> Cs ⊡ Cs,
                  () -> A + A, () -> 2.0 * As, () -> inv(A), () -> det(A), () -> norm(C),
                  () -> m ⋅ m', () -> gradient(norm, As), () -> dotdot(a, Cs, a))
            f() # compile
            @test (@allocated f()) == 0
        end
    end

    @testset "fourth-order symmetry predicates" begin
        for dim in (2, 3)
            # differs only in a component that pairwise triangular checks miss
            A = Tensor{4, dim}((i, j, k, l) -> (i, j, k, l) == (1, 2, 1, 2) ? 1.0 : 0.0)
            @test !isminorsymmetric(A)
            @test_throws InexactError convert(SymmetricTensor{4, dim}, A) # no silent truncation
            B = Tensor{4, dim}((i, j, k, l) -> (i, j, k, l) == (1, 2, 1, 1) ? 1.0 : 0.0)
            @test !ismajorsymmetric(B)
            # actually symmetric tensors still pass
            S = rand(SymmetricTensor{4, dim})
            @test isminorsymmetric(convert(Tensor{4, dim}, S))
            @test ismajorsymmetric(majorsymmetric(rand(Tensor{4, dim})))
        end
    end

    @testset "MixedTensor constructor invariants" begin
        @test_throws ArgumentError MixedTensor{2, Tuple{2, 3}}((1.0, 2.0))
        @test_throws ArgumentError MixedTensor{2, Tuple{2, 3}, Float64}((1.0, 2.0))
        @test_throws ArgumentError MixedTensor{3, Tuple{2, 3}}(ntuple(Float64, 6))
        @test MixedTensor{2, Tuple{2, 3}}(ntuple(Float64, 6)) isa MixedTensor{2, Tuple{2, 3}, Float64, 6}
    end

    @testset "generic-eltype corners" begin
        # tomandel of an integer tensor promotes instead of throwing
        @test tomandel(SymmetricTensor{2, 2}((1, 2, 3))) == [1.0, 3.0, √2 * 2.0]
        @test eltype(tomandel(SymmetricTensor{4, 2}((i, j, k, l) -> 1.0 * (i + j + k + l)))) == Float64
        # 1-D cross has the eltype of the product (matters for units)
        @test cross(Vec{1}((2,)), Vec{1}((3,))) === zero(Vec{3, Int})
        # negative powers of typemin error instead of overflowing
        @test_throws OverflowError rand(Tensor{2, 2})^typemin(Int)
        # Base's one-argument promote contract is not violated
        @test promote(rand(SymmetricTensor{2, 2})) isa Tuple
        @test Tensors.densify(rand(SymmetricTensor{2, 2})) isa Tensor{2, 2}
    end

    @testset "constant tensor-valued functions under gradient" begin
        c12 = Vec{2}((1.0, 2.0))
        @test gradient(x -> c12, rand(Vec{3})) == zero(MixedTensor{2, Tuple{2, 3}})
        @test gradient(x -> c12, rand(Vec{2})) == zero(Tensor{2, 2})
        @test gradient(x -> zero(Tensor{2, 2}), rand(Vec{3})) == zero(MixedTensor{3, Tuple{2, 2, 3}})
    end
end
