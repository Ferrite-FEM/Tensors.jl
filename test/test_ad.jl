const ∇ = ContMechTensors.gradient
const Δ = ContMechTensors.hessian

function Ψ(C, μ, Kb)
    detC = det(C)
    J = sqrt(detC)
    Ĉ = detC^(-1/3)*C
    return 1/2*(μ * (trace(Ĉ)- 3) + Kb*(J-1)^2)
end

function S(C, μ, Kb)
    I = one(C)
    J = sqrt(det(C))
    invC = inv(C)
    return μ * det(C)^(-1/3)*(I - 1/3*trace(C)*invC) + Kb*(J-1)*J*invC
end

const μ = 1e10;
const Kb = 1.66e11;
Ψ(C) = Ψ(C, μ, Kb)
S(C) = S(C, μ, Kb)

@testset "AD" begin
for dim in 1:3
    println("Testing AD for dim = $dim")

    F = one(Tensor{2,dim}) + rand(Tensor{2,dim});
    C = tdot(F);
    C2 = F' ⋅ F;

    @test 2∇(Ψ, C) ≈ S(C)
    @test 2∇(Ψ, C2) ≈ S(C2)

    b = rand(SymmetricTensor{2, dim})
    @test 2 * Δ(Ψ, C) ⊡ b ≈ ∇(S, C) ⊡ b
    @test 2 * Δ(Ψ, C2) ⊡ b ≈ ∇(S, C2) ⊡ b

    @test ∇(Ψ, C) ≈ ∇(Ψ, C2)
    @test ∇(S, C) ⊡ b ≈ ∇(S, C2) ⊡ b
    @test Δ(Ψ, C) ⊡ b ≈ Δ(Ψ, C2) ⊡ b

    for T in (Float32, Float64)
        A = rand(Tensor{2, dim, T})
        B = rand(Tensor{2, dim, T})
        A_sym = rand(SymmetricTensor{2, dim, T})
        B_sym = rand(SymmetricTensor{2, dim, T})
        v = rand(Vec{dim, T})
        II = one(Tensor{4, dim, T})
        II_sym = one(SymmetricTensor{4, dim, T})

        # Gradient of scalars
        @test ∇(norm, v) ≈ v / norm(v)
        @test ∇(norm, A) ≈ A / norm(A)
        @test ∇(norm, A_sym) ≈ A_sym / norm(A_sym)
        @test ∇(v -> 3*v, v) ≈ diagm(Tensor{2, dim}, 3.0)
        # https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)#Derivatives_of_the_invariants_of_a_second-order_tensor
        I1, DI1 = A -> trace(A), A -> one(A)
        I2, DI2 = A -> 1/2 * (trace(A)^2 - trace(A⋅A)), A -> I1(A) * one(A) - A'
        I3, DI3 = A -> det(A), A -> det(A) * inv(A)'

        @test ∇(I1, A) ≈ DI1(A)
        @test ∇(I2, A) ≈ DI2(A)
        @test ∇(I3, A) ≈ DI3(A)
        @test ∇(I1, A_sym) ≈ DI1(A_sym)
        @test ∇(I2, A_sym) ≈ DI2(A_sym)
        @test ∇(I3, A_sym) ≈ DI3(A_sym)

        # Gradient of second order tensors
        @test ∇(identity, A) ≈ II
        @test ∇(identity, A_sym) ≈ II_sym
        @test ∇(transpose, A) ⊡ B ≈ B'
        @test ∇(transpose, A_sym) ⊡ B_sym ≈ B_sym'
        @test ∇(inv, A) ⊡ B ≈ - inv(A) ⋅ B ⋅ inv(A)
        @test ∇(inv, A_sym) ⊡ B_sym ≈ - inv(A_sym) ⋅ B_sym ⋅ inv(A_sym)

        # Hessians of scalars
        @test Δ(norm, A).data ≈ vec(ForwardDiff.hessian(x -> sqrt(sumabs2(x)), A.data))
    end
end
end # testset
