const ∇ = Tensors.gradient
const Δ = Tensors.hessian

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

SUITE["gradient"] = BenchmarkGroup(["ad"])
SUITE["hessian"] = BenchmarkGroup(["ad"])

for dim in (1,2,3)
    for T in (Float64, Float32)
        V2 = tensor_dict[(dim, 2, T)]
        V2 = V2' ⋅ V2
        V2sym = symtensor_dict[(dim, 2, T)]
        V2sym = V2sym' ⋅ V2sym

        SUITE["gradient"]["dim $dim Ψ - sym - $T"] = @benchmarkable ∇(Ψ, $V2sym)
        SUITE["gradient"]["dim $dim Ψ - $T"] = @benchmarkable ∇(Ψ, $V2)
        SUITE["gradient"]["dim $dim Ψ - sym - $T - ana"] = @benchmarkable S($V2sym)
        SUITE["gradient"]["dim $dim Ψ - $T - ana"] = @benchmarkable S($V2)
        SUITE["gradient"]["dim $dim S - sym - $T"] = @benchmarkable ∇(S, $V2sym)
        SUITE["gradient"]["dim $dim S - $T"] = @benchmarkable ∇(S, $V2)

        SUITE["hessian"]["dim $dim Ψ - sym - $T"] = @benchmarkable Δ(Ψ, $V2sym)
        SUITE["hessian"]["dim $dim Ψ - $T"] = @benchmarkable Δ(Ψ, $V2)
    end
end
