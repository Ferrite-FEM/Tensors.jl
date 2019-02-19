# Demos

This section contain a few demos of applying `Tensors` to continuum mechanics.

## Creating the linear elasticity tensor

The linear elasticity tensor $\mathbf{C}$ can be defined from the Lamé parameters $\lambda$ and $\mu$ by the expression

$\mathbf{C}_{ijkl} = \lambda \delta_{ij}\delta_{kl} + \mu(\delta_{ij}\delta_{jl} + \delta_{il}\delta_{jk}),$

where $\delta_{ij} = 1$ if $i = j$ otherwise $0$. It can also be computed in terms of the Young's modulus $E$ and Poisson's ratio $\nu$ by the conversion formulas $\lambda = E\nu / [(1 + \nu)(1 - 2\nu)]$ and $\mu = E / [2(1 + \nu)]$

The code below creates the elasticity tensor for given parameters $E$ and $\nu$ and dimension $\texttt{dim}$. Note the similarity between the mathematical formula and the code.

```julia
using Tensors
E = 200e9
ν = 0.3
dim = 2
λ = E*ν / ((1 + ν) * (1 - 2ν))
μ = E / (2(1 + ν))
δ(i,j) = i == j ? 1.0 : 0.0
f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

C = SymmetricTensor{4, dim}(f)
```

## Nonlinear elasticity material

For a deformation gradient $\mathbf{F} = \mathbf{I} + \nabla \otimes \mathbf{u}$, where $\mathbf{u}$ is the deformation from the reference to the current configuration, the right Cauchy-Green deformation tensor is defined by $\mathbf{C} = \mathbf{F}^T \cdot \mathbf{F}$. The Second Piola Krichoff stress tensor $\mathbf{S}$ is derived from the Helmholtz free energy $\Psi$ by the relation $\mathbf{S} = 2 \frac{\partial \Psi}{\partial \mathbf{C}}$.

We can define potential energy of the material as

$\Psi(\mathbf{C}) = 1/2 \mu (\mathrm{tr}(\hat{\mathbf{C}}) - 3) + K_b(J-1)^2,$

where $\hat{\mathbf{C}} = \mathrm{det}(\mathbf{C})^{-1/3} \mathbf{C}$ and $J = \det(\mathbf{F}) = \sqrt{\det(\mathbf{C})}$ and the shear and bulk modulus are given by $\mu$ and $K_b$ respectively.

This free energy function can be implemented in `Tensors` as:

```julia
function Ψ(C, μ, Kb)
    detC = det(C)
    J = sqrt(detC)
    Ĉ = detC^(-1/3)*C
    return 1/2*(μ * (tr(Ĉ)- 3) + Kb*(J-1)^2)
end
```

The analytical expression for the Second Piola Kirchoff tensor is

$ \mathbf{S} = \mu \det(\mathbf{C})^{-1/3}(\mathbf{I} - 1/3 \mathrm{tr}(\mathbf{C})\mathbf{C}^{-1}) + K_b(J-1)J\mathbf{C}^{-1}
$

which can be implemented by the function

```julia
function S(C, μ, Kb)
    I = one(C)
    J = sqrt(det(C))
    invC = inv(C)
    return μ * det(C)^(-1/3)*(I - 1/3*tr(C)*invC) + Kb*(J-1)*J*invC
end
```

### Automatic differentiation

For some material models it can be cumbersome to compute the analytical expression for the Second Piola Kirchoff tensor. We can then instead use Automatic Differentiation (AD). Below is an example which computes the Second Piola Kirchoff tensor using AD and compares it to the analytical answer.

```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
    E = 200e9
    ν = 0.3
    dim = 2
    λ = E*ν / ((1 + ν) * (1 - 2ν))
    μ = E / (2(1 + ν))
    δ(i,j) = i == j ? 1.0 : 0.0
    f = (i,j,k,l) -> λ*δ(i,j)*δ(k,l) + μ*(δ(i,k)*δ(j,l) + δ(i,l)*δ(j,k))

    C = SymmetricTensor{4, dim}(f)

    function Ψ(C, μ, Kb)
        detC = det(C)
        J = sqrt(detC)
        Ĉ = detC^(-1/3)*C
        return 1/2*(μ * (tr(Ĉ)- 3) + Kb*(J-1)^2)
    end

    function S(C, μ, Kb)
        I = one(C)
        J = sqrt(det(C))
        invC = inv(C)
        return μ * det(C)^(-1/3)*(I - 1/3*tr(C)*invC) + Kb*(J-1)*J*invC
    end
end
```

```jldoctest
julia> μ = 1e10;

julia> Kb = 1.66e11;

julia> F = one(Tensor{2,3}) + rand(Tensor{2,3});

julia> C = tdot(F);

julia> S_AD = 2 * gradient(C -> Ψ(C, μ, Kb), C)
3×3 SymmetricTensor{2,3,Float64,6}:
  4.30534e11  -2.30282e11  -8.52861e10
 -2.30282e11   4.38793e11  -2.64481e11
 -8.52861e10  -2.64481e11   7.85515e11

julia> S(C, μ, Kb)
3×3 SymmetricTensor{2,3,Float64,6}:
  4.30534e11  -2.30282e11  -8.52861e10
 -2.30282e11   4.38793e11  -2.64481e11
 -8.52861e10  -2.64481e11   7.85515e11
```
