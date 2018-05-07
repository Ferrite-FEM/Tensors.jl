import ForwardDiff: valtype
const ∇ = Tensors.gradient
const Δ = Tensors.hessian

function Ψ(C, μ, Kb)
    T = ForwardDiff.valtype(eltype(C))
    detC = det(C)
    J = sqrt(detC)
    Ĉ = detC^(convert(T, -1 / 3)) * C
    return (convert(T, μ) * (tr(Ĉ) - 3) / 2 + convert(T, Kb) * (J - 1)^2)
end

function S(C, μ, Kb)
    T = ForwardDiff.valtype(eltype(C))
    I = one(C)
    J = sqrt(det(C))
    invC = inv(C)
    return  convert(T, μ) * det(C)^(convert(T, -1/3)) * (I - tr(C) * invC / 3) + convert(T, Kb) * (J - 1) * J * invC
end

const μ = 1e10;
const Kb = 1.66e11;

Ψ(C) = Ψ(C, μ, Kb)
S(C) = S(C, μ, Kb)

SUITE["gradient"] = BenchmarkGroup(["ad"])
SUITE["hessian"] = BenchmarkGroup(["ad"])

for dim in (ALL_DIMENSIONS ? (1,2,3) : (3,))
    for T in (Float64, Float32)
        V2 = tensor_dict[(dim, 2, T)]
        V2 = V2' ⋅ V2
        V2sym = symtensor_dict[(dim, 2, T)]
        V2sym = V2sym' ⋅ V2sym

        @assert eltype(Ψ(V2)) == T
        @assert eltype(S(V2)) == T
        @assert eltype(∇(Ψ, V2)) == T
        @assert eltype(Δ(Ψ, V2)) == T

        SUITE["gradient"]["∇Ψ(SymmetricTensor{2, $dim, $T})"]       = @benchmarkable ∇(Ψ, $V2sym)
        SUITE["gradient"]["∇Ψ(Tensor{2, $dim, $T})"]                = @benchmarkable ∇(Ψ, $V2)
        SUITE["gradient"]["∇Ψ(SymmetricTensor{2, $dim, $T}) - ana"] = @benchmarkable S($V2sym)
        SUITE["gradient"]["∇Ψ(Tensor{2, $dim, $T}) - ana"]          = @benchmarkable S($V2)
        SUITE["gradient"]["∇S(SymmetricTensor{2, $dim, $T})"]       = @benchmarkable ∇(S, $V2sym)
        SUITE["gradient"]["∇S(Tensor{2, $dim, $T})"]                = @benchmarkable ∇(S, $V2)

        SUITE["hessian"]["ΔΨ(SymmetricTensor{2, $dim, $T})"] = @benchmarkable Δ(Ψ, $V2sym)
        SUITE["hessian"]["ΔΨ(Tensor{2, $dim, $T})"]          = @benchmarkable Δ(Ψ, $V2)
    end
end
