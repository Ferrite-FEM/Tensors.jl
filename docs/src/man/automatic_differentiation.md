```@meta
DocTestSetup = quote
    using Random
    Random.seed!(1234)
    using Tensors
end
```

# Automatic Differentiation

```@index
Pages = ["automatic_differentiation.md"]
```

`Tensors` supports forward mode automatic differentiation (AD) of tensorial functions to compute first order derivatives (gradients) and second order derivatives (Hessians).
It does this by exploiting the `Dual` number defined in `ForwardDiff.jl`.
While `ForwardDiff.jl` can itself be used to differentiate tensor functions it is a bit awkward because `ForwardDiff.jl` is written to work with standard Julia `Array`s. One therefore has to send the input argument as an `Array` to `ForwardDiff.jl`, convert it to a `Tensor` and then convert the output `Array` to a `Tensor` again. This can also be inefficient since these `Array`s are allocated on the heap so one needs to preallocate which can be annoying.

Instead, it is simpler to use `Tensors` own AD API to do the differentiation. This does not require any conversions and everything will be stack allocated so there is no need to preallocate.

The API for AD in `Tensors` is `gradient(f, A)` and `hessian(f, A)` where `f` is a function and `A` is a first or second order tensor. For `gradient` the function can return a scalar, vector (in case the input is a vector) or a second order tensor. For `hessian` the function should return a scalar.

When evaluating the function with dual numbers, the value (value and gradient in the case of hessian) is obtained automatically, along with the gradient. To obtain the lower order results `gradient` and `hessian` accepts a third arguement, a `Symbol`. Note that the symbol is only used to dispatch to the correct function, and thus it can be any symbol. In the examples the symbol `:all` is used to obtain all the lower order derivatives and values.

```@docs
gradient
hessian
divergence
curl
laplace
```

## Examples

We here give a few examples of differentiating various functions and compare with the analytical solution.

### Norm of a vector

```math
f(\mathbf{x}) = |\mathbf{x}| \quad \Rightarrow \quad \partial f / \partial \mathbf{x} = \mathbf{x} / |\mathbf{x}|
```

```jldoctest
julia> x = rand(Vec{2});

julia> gradient(norm, x)
2-element Tensor{1,2,Float64,2}:
 0.6103600560550116
 0.7921241076829584

julia> x / norm(x)
2-element Tensor{1,2,Float64,2}:
 0.6103600560550116
 0.7921241076829584
```

### Determinant of a second order symmetric tensor

```math
f(\mathbf{A}) = \det \mathbf{A} \quad \Rightarrow \quad \partial f / \partial \mathbf{A} = \mathbf{A}^{-T} \det \mathbf{A}
```

```jldoctest
julia> A = rand(SymmetricTensor{2,2});

julia> gradient(det, A)
2×2 SymmetricTensor{2,2,Float64,3}:
  0.566237  -0.766797
 -0.766797   0.590845

julia> inv(A)' * det(A)
2×2 SymmetricTensor{2,2,Float64,3}:
  0.566237  -0.766797
 -0.766797   0.590845
```

### Hessian of a quadratic potential

```math
\psi(\mathbf{e}) = 1/2 \mathbf{e} : \mathsf{E} : \mathbf{e} \quad \Rightarrow \quad \partial \psi / (\partial \mathbf{e} \otimes \partial \mathbf{e}) = \mathsf{E}^\text{sym}
```

where ``\mathsf{E}^\text{sym}`` is the major symmetric part of ``\mathsf{E}``.

```jldoctest
julia> E = rand(Tensor{4,2});

julia> ψ(ϵ) = 1/2 * ϵ ⊡ E ⊡ ϵ;

julia> E_sym = hessian(ψ, rand(Tensor{2,2}));

julia> norm(majorsymmetric(E) - E_sym)
0.0
```
