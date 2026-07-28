# Internals: the code-generation engine

!!! warning "Internal API"
    Everything on this page is internal and may change between minor versions.
    It is documented for contributors, not for users of the package.

All products and contractions in `Tensors` are defined through a single
code-generation engine in `src/einsum.jl`. An operation is declared with the
internal `Tensors.@tensorop` macro in index notation:

```julia
@tensorop function dcontract(A::FourthOrderTensor, B::SecondOrderTensor)
    @muladd C[i, j] = A[i, j, k, l] * B[k, l]
end
```

The left-hand side names the output indices (`C = ...` declares a scalar
output); an index that appears in two arguments is summed over; wrapping the
assignment in `@muladd` makes the summation accumulate with `muladd`. The
declaration expands to a single `@generated` method with the written
signature.

Products of more than two tensors are supported as well — each index must
appear either once (an output index) or in exactly two of the arguments (a
summation index). For example, `dotdot` is declared as

```julia
@tensorop function dotdot(v1::AbstractTensor{1}, S::FourthOrderTensor, v2::AbstractTensor{1})
    @muladd C[i, j] = v1[k] * S[i, k, j, l] * v2[l]
end
```

(SIMD lowering applies to two-argument operations; operations with more
arguments use the scalar lowering.)

## What the generator does

When the method is compiled for concrete argument types, the planner
(`Tensors.einsum_expr`) sees the actual tensor kinds — `Tensor`,
`SymmetricTensor` or `MixedTensor`, their dimensions and element types — and
emits a flat component-tuple expression for exactly that combination:

* Data is read through `Tensors.compute_index`, so symmetric arguments are
  indexed directly in their packed storage. Products that are equal by
  symmetry collapse into integer prefactors (`2 * a * b` — exact for
  floating point).
* The output type is computed from the index structure: an output index pair
  that provably carries symmetry from a symmetric argument gives a
  `SymmetricTensor`; differing output dimensions give a `MixedTensor`, which
  collapses to a `Tensor` when all dimensions agree.
* Index dimensions that cannot agree produce code that throws a runtime
  `DimensionMismatch` (never an error inside the generator).
* Each component is a left-fold sum-of-products starting from the first
  product (no `zero(T)` seed), which is the layout LLVM vectorizes best and
  matches the pre-rewrite package expression-for-expression.

## SIMD lowering

For same-eltype hardware floats (`Float16/32/64`), `Tensors.try_simd_expr`
(`src/simd_lowering.jl`) inspects the planned products for packed-space
column structure: within an output column of height `m`, every term must read
`m` contiguous elements of one argument scaled by a single element of the
other. When the plan has that shape, the generator emits `SIMD.Vec`
column-load/`muladd` kernels instead of scalar code — the same kernels the
old hand-written `simd.jl` contained, now derived mechanically. The numerics
contract: each (operation, eltype) computes exactly what the pre-rewrite
package computed. For operations declared with `@muladd` (the `dcontract`
family) the SIMD and scalar lowerings additionally agree bit-for-bit; for
operations declared without it (plain `dot`, `otimes`) the SIMD kernels
always chain `muladd`s, so the result may differ from the scalar path's
`a*b + c*d` in the last ulp where fma contracts, and scalar-output kernels
use a horizontal vector sum. Plans without column structure (and all other
element types) fall back to the scalar lowering. Element-wise fast paths
(`+`, `-`, scalar `*` and `/`, `norm`) live in the same file.

## Adding an operation

1. Declare it with `@tensorop` next to the related operations in
   `src/tensor_products.jl`. One declaration covers all tensor kinds,
   dimensions and element types for that order combination.
2. Operations that are not index-notation shaped (determinants, inverses,
   rotations, ...) are written by hand as before, usually over the
   component-map layer in `src/maps.jl` (`apply_all`, `_map`,
   `component_expr`).
3. Add cases to `test/test_einsum.jl`: it checks new combinations against a
   brute-force `Array` reference in exact integer arithmetic, verifies output
   types, and asserts SIMD/scalar agreement.

The engine-level regression tests in `test/test_einsum.jl` are the contract
for all of the above (plan structure, storage order, symmetric
multiplicities, error behavior, and the `@tensorop` expansion itself).

## Internal reference

```@docs
Tensors.@tensorop
Tensors.einsum_expr
Tensors._extract_value
Tensors._insert_gradient
```
