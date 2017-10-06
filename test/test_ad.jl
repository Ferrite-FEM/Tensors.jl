using Tensors: ∇, ∇∇, Δ

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

@testsection "AD" begin
    for dim in 1:3
        @testsection "dim $dim" begin
        F = one(Tensor{2,dim}) + rand(Tensor{2,dim});
        C = tdot(F);
        C2 = F' ⋅ F;

        @test 2(@inferred ∇(Ψ, C))::typeof(C) ≈ S(C)
        @test 2(@inferred ∇(Ψ, C2))::typeof(C2) ≈ S(C2)

        b = rand(SymmetricTensor{2, dim})
        @test 2 * (@inferred ∇∇(Ψ, C))::SymmetricTensor{4, dim} ⊡ b ≈ ∇(S, C) ⊡ b
        @test 2 * (@inferred ∇∇(Ψ, C2))::Tensor{4, dim} ⊡ b ≈ ∇(S, C2) ⊡ b

        @test ∇(Ψ, C) ≈ ∇(Ψ, C2)
        @test ∇(S, C) ⊡ b ≈ ∇(S, C2) ⊡ b
        @test ∇∇(Ψ, C) ⊡ b ≈ ∇∇(Ψ, C2) ⊡ b

        for T in (Float32, Float64)
            srand(1234) # needed for getting "good" tensors for calculating det and friends
            A = rand(Tensor{2, dim, T})
            B = rand(Tensor{2, dim, T})
            A_sym = rand(SymmetricTensor{2, dim, T})
            B_sym = rand(SymmetricTensor{2, dim, T})
            v = rand(Vec{dim, T})
            II = one(Tensor{4, dim, T})
            II_sym = one(SymmetricTensor{4, dim, T})

            # Gradient of scalars
            @testsection "scalar grad" begin
                @test (@inferred ∇(norm, v))::typeof(v) ≈ ((@inferred ∇(norm, v, :all))[1])::typeof(v) ≈ v / norm(v)
                @test (∇(norm, v, :all)[2])::T ≈ norm(v)
                @test (@inferred ∇(norm, A))::typeof(A) ≈ ((@inferred ∇(norm, A, :all))[1])::typeof(A) ≈ A / norm(A)
                @test (∇(norm, A, :all)[2])::T ≈ norm(A)
                @test (@inferred ∇(norm, A_sym))::typeof(A_sym) ≈ ((@inferred ∇(norm, A_sym, :all))[1])::typeof(A_sym) ≈ A_sym / norm(A_sym)
                @test (∇(norm, A_sym, :all)[2])::T ≈ norm(A_sym)

                @test ∇(v -> 3*v, v) ≈ ∇(v -> 3*v, v, :all)[1] ≈ diagm(Tensor{2, dim}, 3.0)
                @test ∇(v -> 3*v, v, :all)[2] ≈ 3*v
                # function does not return dual
                @test ∇(A -> T(1), A)::typeof(A) ≈ ∇(A -> T(1), A, :all)[1] ≈ 0*A
                @test ∇(A -> T(1), A, :all)[2] == 1
                @test ∇(A -> T(1), A_sym)::typeof(A_sym) ≈ ∇(A -> T(1), A, :all)[1] ≈ 0*A_sym
                @test ∇(A -> T(1), A_sym, :all)[2] == 1
            end

            @testsection "2nd tensor grad" begin
            # Gradient of second order tensors
            # https://en.wikipedia.org/wiki/Tensor_derivative_(continuum_mechanics)#Derivatives_of_the_invariants_of_a_second-order_tensor
                I1, DI1 = A -> trace(A), A -> one(A)
                I2, DI2 = A -> (trace(A)^2 - trace(A⋅A)) / 2, A -> I1(A) * one(A) - A'
                I3, DI3 = A -> det(A), A -> det(A) * inv(A)'

                @test (@inferred ∇(I1, A))::typeof(A) ≈ ((@inferred ∇(I1, A, :all))[1])::typeof(A) ≈ DI1(A)
                @test (∇(I1, A, :all)[2])::T ≈ I1(A)
                @test (@inferred ∇(I2, A))::typeof(A) ≈ ((@inferred ∇(I2, A, :all))[1])::typeof(A) ≈ DI2(A)
                @test (∇(I2, A, :all)[2])::T ≈ I2(A)
                @test (@inferred ∇(I3, A))::typeof(A) ≈ ((@inferred ∇(I3, A, :all))[1])::typeof(A) ≈ DI3(A)
                @test (∇(I3, A, :all)[2])::T ≈ I3(A)
                @test (@inferred ∇(I1, A_sym))::typeof(A_sym) ≈ ((@inferred ∇(I1, A_sym, :all))[1])::typeof(A_sym) ≈ DI1(A_sym)
                @test (∇(I1, A_sym, :all)[2])::T ≈ I1(A_sym)
                @test (@inferred ∇(I2, A_sym))::typeof(A_sym) ≈ ((@inferred ∇(I2, A_sym, :all))[1])::typeof(A_sym) ≈ DI2(A_sym)
                @test (∇(I2, A_sym, :all)[2])::T ≈ I2(A_sym)
                @test (@inferred ∇(I3, A_sym))::typeof(A_sym) ≈ ((@inferred ∇(I3, A_sym, :all))[1])::typeof(A_sym) ≈ DI3(A_sym)
                @test (∇(I3, A_sym, :all)[2])::T ≈ I3(A_sym)

                @test (@inferred ∇(identity, A))::Tensor{4, dim, T} ≈ ((@inferred ∇(identity, A, :all))[1])::Tensor{4, dim, T} ≈ II
                @test (∇(identity, A, :all)[2])::typeof(A) ≈ A
                @test (@inferred ∇(identity, A_sym))::SymmetricTensor{4, dim, T} ≈ ((@inferred ∇(identity, A_sym, :all))[1])::SymmetricTensor{4, dim, T} ≈ II_sym
                @test (∇(identity, A_sym, :all)[2])::typeof(A_sym) ≈ A_sym
                @test (@inferred ∇(transpose, A))::Tensor{4, dim, T} ⊡ B ≈ ((@inferred ∇(transpose, A, :all))[1])::Tensor{4, dim, T} ⊡ B ≈ B'
                @test (∇(transpose, A, :all)[2])::typeof(A) ≈ A'
                @test (@inferred ∇(transpose, A_sym))::SymmetricTensor{4, dim, T} ⊡ B_sym ≈ ((@inferred ∇(transpose, A_sym, :all))[1])::SymmetricTensor{4, dim, T} ⊡ B_sym ≈ B_sym'
                @test (∇(transpose, A_sym, :all)[2])::typeof(A_sym) ≈ A_sym'
                @test (@inferred ∇(inv, A))::Tensor{4, dim, T} ⊡ B ≈ ((@inferred ∇(inv, A, :all))[1])::Tensor{4, dim, T} ⊡ B ≈ - inv(A) ⋅ B ⋅ inv(A)
                @test (∇(inv, A, :all)[2])::typeof(A) ≈ inv(A)
                @test (@inferred ∇(inv, A_sym))::SymmetricTensor{4, dim, T} ⊡ B_sym ≈ ((@inferred ∇(inv, A_sym, :all))[1])::SymmetricTensor{4, dim, T} ⊡ B_sym ≈ - inv(A_sym) ⋅ B_sym ⋅ inv(A_sym)
                @test (∇(inv, A_sym, :all)[2])::typeof(A_sym) ≈ inv(A_sym)
                # function does not return dual
                @test ∇(A -> B, A) ≈ ∇(A -> B, A, :all)[1] ≈ zero(Tensor{4, dim, T})
                @test ∇(A -> B, A, :all)[2] ≈ B
                @test isa(∇(A -> B, A), Tensor{4, dim, T})
            end

            # Hessians of scalars
            @testsection "hessian" begin
                @test (@inferred ∇∇(norm, A))::Tensor{4, dim, T} ≈ ((@inferred ∇∇(norm, A, :all))[1])::Tensor{4, dim, T} ≈ reshape(ForwardDiff.hessian(x -> sqrt(sum(abs2, x)), A), (dim, dim, dim, dim))
                @test (∇∇(norm, A, :all)[2])::typeof(A) ≈ reshape(ForwardDiff.gradient(x -> sqrt(sum(abs2, x)), A), (dim, dim))
                @test (∇∇(norm, A, :all)[3])::T ≈ norm(A)
                @test (@inferred ∇∇(norm, A_sym))::SymmetricTensor{4, dim, T} ≈ ((@inferred ∇∇(norm, A_sym, :all))[1])::SymmetricTensor{4, dim, T}
                @test (∇∇(norm, A_sym, :all)[2])::typeof(A_sym) ≈ ∇(norm, A_sym)
                @test (∇∇(norm, A_sym, :all)[3])::T ≈ norm(A_sym)
                # function does not return dual
                @test ∇∇(A -> T(1), A)::Tensor{4, dim, T} ≈ ∇∇(A -> T(1), A, :all)[1] ≈ 0*II
                @test ∇∇(A -> T(1), A, :all)[2] ≈ 0*A
                @test ∇∇(A -> T(1), A, :all)[3] == T(1)
            end
            end # loop T
        end # testsection
    end # loop dim

    @testsection "vector calculus identities" begin
        φ(x) = norm(x)^4
        ϕ(x) = sum(x)
        A(x) = Vec{3}((x[1]*x[2]^3*x[3], x[1]*x[2]*x[3]^3, x[1]^3*x[2]*x[3]))
        B(x) = Vec{3}((x[1]*x[1], x[1]*x[2], x[1]*x[3]))
        for T in (Float32, Float64)
            x = rand(Vec{3, T})
            # gradient
            @test gradient(x -> φ(x) + ϕ(x), x) ≈ gradient(φ, x) + gradient(ϕ, x)
            @test gradient(x -> φ(x) * ϕ(x), x) ≈ φ(x) * gradient(ϕ, x) + gradient(φ, x) * ϕ(x)
            @test gradient(x -> A(x) ⋅ B(x), x) ≈ gradient(B, x)⋅A(x) + gradient(A, x)⋅B(x) + A(x)×curl(B, x) + B(x)×curl(A, x)
            # divergence
            @test div(x -> A(x) + B(x), x) ≈ div(A, x) + div(B, x)
            @test div(x -> φ(x) * A(x), x) ≈ φ(x)*div(A, x) + gradient(φ, x)⋅A(x)
            @test div(x -> A(x) × B(x), x) ≈ B(x)⋅curl(A, x) - A(x)⋅curl(B, x)
            # curl
            @test curl(x -> A(x) + B(x), x) ≈ curl(A, x) + curl(B, x)
            @test curl(x -> φ(x) * A(x), x) ≈ φ(x)*curl(A, x) + gradient(φ, x)×A(x)
            @test curl(x -> A(x) × B(x), x) ≈ A(x)*div(B, x) - B(x)*div(A, x) + gradient(A, x)⋅B(x) - gradient(B, x)⋅A(x)
            # second derivatives
            @test div(x -> curl(A, x), x) ≈ 0 atol = eps(T)
            @test curl(x -> gradient(φ, x), x) ≈ zero(Vec{3}) atol = 10eps(T)
            @test div(x -> gradient(φ, x), x) ≈ laplace(φ, x)
            @test gradient(x -> div(A, x), x) ≈ curl(x -> curl(A, x), x) + laplace.(A, x)
            @test div(x -> ϕ(x)*gradient(φ, x), x) ≈ ϕ(x)*laplace(φ, x) + gradient(ϕ, x)⋅gradient(φ, x)
            @test div(x -> φ(x)*gradient(ϕ, x) - gradient(φ, x)*ϕ(x), x) ≈ φ(x)*laplace(ϕ, x) - laplace(φ, x)*ϕ(x)
            @test laplace(x -> ϕ(x)*φ(x), x) ≈ laplace(ϕ, x)*φ(x) + 2*gradient(ϕ, x)⋅gradient(φ, x) + ϕ(x)*laplace(φ, x)
            @test laplace.(x -> φ(x)*A(x), x) ≈ A(x)*laplace(φ, x) + 2*(gradient(A, x)⋅gradient(φ, x)) + φ(x)*laplace.(A, x)
            @test laplace(x -> A(x)⋅B(x), x) ≈ A(x)⋅laplace.(B, x) - B(x)⋅laplace.(A, x) + 2*div(x -> gradient(A, x)⋅B(x) + B(x)×curl(A, x), x)
            # third derivatives
            @test laplace.(x -> gradient(φ, x), x) ≈ gradient(x -> div(x -> gradient(φ, x), x), x) ≈ gradient(x -> laplace(φ, x), x)
            @test laplace(x -> div(A, x), x) ≈ div(x -> gradient(x->div(A, x), x), x) ≈ div(x -> laplace.(A, x), x)
            @test laplace.(x -> curl(A, x), x) ≈ -curl(x -> curl(x -> curl(A, x), x), x) ≈ curl(x -> laplace.(A, x), x)
        end # loop T
    end # testsection
end # testsection
