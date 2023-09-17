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
2-element Vec{2, Float64}:
 0.6103600560550116
 0.7921241076829584

julia> x / norm(x)
2-element Vec{2, Float64}:
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
2×2 SymmetricTensor{2, 2, Float64, 3}:
  0.566237  -0.766797
 -0.766797   0.590845

julia> inv(A)' * det(A)
2×2 SymmetricTensor{2, 2, Float64, 3}:
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

## Differentiating mutating functions
Some applications require the derivative of the output of a function `f(x,s)`, 
wrt. `x`, where `f` also mutates `s`. 
In these cases, we don't want the derivative of `s` wrt. `x`, and 
the value to be set in `s` should only be the value and not be the dual part.
For scalars, `ForwardDiff.jl` provides `ForwardDiff.value`,
and for tensors, `Tensors.jl` provides `Tensors.extract_value`. 

```@docs
Tensors.extract_value
```

A simple example of the use-case is 
```@example
function mutating_fun(x::Vec, state::Vector)
    state[1] = Tensors.extract_value(x)
    return x
end

x = rand(Vec{2}); state = zeros(Vec{2}, 1)
gradient(a -> mutating_fun(a, state, true), x)
# Check that it got correctly modified by the extracted value
state[1] == x 
```

## Inserting a known derivative
When conditionals are used in a function evaluation, automatic differentiation 
may yield the wrong result. Consider, the simplified example of the function 
`f(x) = is_zero(x) ? zero(x) : sin(x)`. If evaluated at `x=0`, the returning 
of `zero(x)` gives a zero derivative because `zero(x)` is constant, while the 
correct value is 1. In such cases, it is possible to insert a known 
derivative of a function which is part of a larger function to be 
automatically differentiated.

Another use case is when the analytical derivative can be computed much more 
efficiently than the automatically differentiatiated derivative.

```@docs
@implement_gradient
```

### Example
Lets consider the function ``h(\mathbf{f}(\mathbf{g}(\mathbf{x})))`` 
where `h(x)=norm(x)`, `f(x)=x ⋅ x`, and `g(x)=dev(x)`. For `f(x)` we 
then have the analytical derivative 
```math
\frac{\partial f_{ij}}{\partial x_{kl}} = \delta_{ik} x_{lj} + x_{ik} \delta_{jl}
```
which we can insert into our known analytical derivative using the
 `@implement_gradient` macro. Below, we compare with the result when 
 the full derivative is calculated using automatic differentiation.

```jldoctest
# Define functions
h(x) = norm(x)
f1(x) = x ⋅ x
f2(x) = f1(x)
g(x) = dev(x)

# Define composed functions
cfun1(x) = h(f1(g(x)))
cfun2(x) = h(f2(g(x)))

# Define known derivative
function df2dx(x::Tensor{2,dim}) where{dim}
    println("Hello from df2dx") # Show that df2dx is called
    fval = f2(x)
    I2 = one(Tensor{2,dim})
    dfdx_val = otimesu(I2, transpose(x)) + otimesu(x, I2)
    return fval, dfdx_val
end

# Implement known derivative
@implement_gradient f2 df2dx

# Calculate gradients
x = rand(Tensor{2,2})

gradient(cfun1, x) ≈ gradient(cfun2, x)

# output
Hello from df2dx
true
```