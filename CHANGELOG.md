# Tensors changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [v1.18.0] (unreleased)

### Changed
- The internals have been rewritten around a single index-notation code
  generator (`@tensorop`/`einsum_expr` in `src/einsum.jl`): all contractions and
  products are declared in index notation (e.g. `C[i,j] = A[i,k] * B[k,j]`) and
  one `@generated` method per declaration covers `Tensor`, `SymmetricTensor`
  and `MixedTensor` arguments of all dimensions and element types. The
  hand-written per-operation methods (`src/utilities.jl`) and the hand-unrolled
  SIMD kernels (`src/simd.jl`) are replaced; SIMD code is now generated from
  the same operation descriptions (`src/simd_lowering.jl`). Public API, storage
  layout, values, and performance are unchanged.

### Added
- Previously missing contraction combinations now exist: `dot` for
  (3rd, 3rd)-order — giving a 4th-order result ([#227]) — as well as
  (1st, 4th)/(4th, 1st)-order, and `otimes` for (1st, 3rd)/(3rd, 1st)-order.
- Broadcasting where the only container arguments are tensors of one shape now
  returns a tensor instead of silently materializing an `Array` ([#223]), e.g.
  `S .+ 1.0` gives a `Tensor{2}` (a `SymmetricTensor` densifies, since a general
  broadcast function need not preserve symmetry). Broadcasts mixing tensors with
  ordinary arrays, or with non-`Number` results, still return `Array`s.
- Non-literal integer powers of second-order tensors, `S^(n::Integer)` ([#239]).
- `propagate_gradient(f_dfdx, x, args...)` (the building block of
  `@implement_gradient`) is now public and supports passing through additional
  non-differentiated arguments ([#197]), and can be used to define
  analytical-derivative methods with precise signatures ([#179]).
- `extract_value(x)` is now public: strips one level of `ForwardDiff.Dual` from
  a number or tensor, e.g. for storing state variables under AD ([#208]).
- `promote`/`convert` between `MixedTensor`s that differ only in element type.
- The internal `@tensorop` code generator supports products of more than two
  tensors; `dotdot` is now declared that way and accepts any fourth-order
  tensor (previously `SymmetricTensor{4}` only).
- `dotdot` gained two bilinear-form methods: `dotdot(a, S, b)` for
  ``a_i S_{ij} b_j`` with a second-order `S`, and `dotdot(A, C, B)` for
  ``A_{ij} C_{ijkl} B_{kl}``.
- `propagate_gradient` supports several active arguments,
  `propagate_gradient(f_dfdx, Val((i, j)), args...)`, for analytical
  derivatives of functions where more than one argument depends on the
  differentiated variable; the chain-rule contributions are summed, and
  mixing dual numbers from different differentiation calls is an error.

### Changed (AD internals)
- Analytical-gradient insertion (`propagate_gradient`/`@implement_gradient`) no
  longer reconstructs the differentiation input from the `ForwardDiff.Tag` type
  parameters; the Jacobian is applied directly to the incoming partial lanes.
  Analytical gradients therefore now compose with any outer differentiation
  context: plain `ForwardDiff` calls, nested duals (e.g. `hessian` through an
  analytical gradient), and tags not created by Tensors ([#179]).

### Bugfixes
- `zero`, `one`, `ones`, `rand`, `randn` of a `MixedTensor` now return a
  `MixedTensor` instead of falling back to `Array` ([#245]).
- Contractions between `MixedTensor`s with mismatching index dimensions now
  throw a `DimensionMismatch` error instead of an internal generator error.
- `isminorsymmetric` and `ismajorsymmetric` previously missed certain
  asymmetric components (e.g. a tensor with `A[1,2,1,2] != A[2,1,2,1]` was
  reported minor symmetric), which could silently corrupt data through
  `convert(SymmetricTensor{4,dim}, A)`. Both predicates now check every
  component.
- `MixedTensor{order, dims}` constructors validate that the number of
  dimensions matches the order and that the data has the right number of
  components (previously a too-short tuple was accepted and later read out
  of bounds).
- Constant tensor-valued functions with an output shape differing from the
  input no longer error under `gradient` (a zero gradient of the appropriate
  shape is returned).
- `tomandel`/`tovoigt` with `offdiagscale` promote the element type of the
  returned array, so integer tensors no longer throw `InexactError`.
- The cross product of two `Vec{1}` returns the element type of the product
  (relevant for `Unitful`-style quantities).
- `S^(p::Integer)` with `p == typemin(Int)` throws `OverflowError` instead of
  recursing.
- The one-argument `Base.promote(::AbstractTensor)` method is removed: it
  returned a bare (densified) tensor, violating `promote`'s contract of
  returning a tuple. Use `Tensors.densify` for the old behavior.

## [v1.17.0]

### Added
- Support for `MixedTensors` and enhanced support for 3rd order tensors ([#236])
- Internal code-generation generalization for tensor products ([#233])

### Bugfixes
 - Throw error if trying to call fallback dot method for AbstractArray ([#228])
 - Fix that `curl(f, v::Vec{2})` calls `f(::Vec{3, <:Dual})` instead of `f(::Vec{2, <:Dual})`([#222])

### Misc
 - This and later versions of Tensors require Julia 1.10. ([#238], [#240])

## [v1.16.2]

### Misc

 - Relax compat for ForwardDiff to allow version 1 ([#225]).

## [v1.16.1]

### Bugfixes

 - Fix that `tovoigt!(::Vector{TA}, ::AbstractTensor{order,dim,TB})` didn't work after v1.15 unless `TA==TB` ([#212]).

## [v1.16.0]

### Added
 - Partial support for 3rd order Tensors ([#205]):
    * All construction methods, e.g. `zero(Tensor{3})`, `rand(Tensor{3})`, `Tensor{3}((i,j,k)->f(i,j,k))`
    * Gradient of 2nd order tensor wrt. vector
    * `rotate(::Tensor{3})`
    * `dcontract(::Tensor{D1}, ::Tensor{D2})` for (D1,D2) in ((2,3), (3,2), (3,4), (4,3))
    * `otimes(::Vec, ::SecondOrderTensor)` and `otimes(::SecondOrderTensor, ::Vec)`
    * `dot(::Tensor{D1}, ::Tensor{D2})` for (D1,D2) in ((3,1), (1,3), (2,3), (3,2))


<!-- Links generated by Changelog.jl -->

[v1.16.0]: https://github.com/Ferrite-FEM/Tensors.jl/releases/tag/v1.16.0
[v1.16.1]: https://github.com/Ferrite-FEM/Tensors.jl/releases/tag/v1.16.1
[v1.16.2]: https://github.com/Ferrite-FEM/Tensors.jl/releases/tag/v1.16.2
[v1.17.0]: https://github.com/Ferrite-FEM/Tensors.jl/releases/tag/v1.17.0
[#205]: https://github.com/Ferrite-FEM/Tensors.jl/issues/205
[#212]: https://github.com/Ferrite-FEM/Tensors.jl/issues/212
[#222]: https://github.com/Ferrite-FEM/Tensors.jl/issues/222
[#225]: https://github.com/Ferrite-FEM/Tensors.jl/issues/225
[#228]: https://github.com/Ferrite-FEM/Tensors.jl/issues/228
[#233]: https://github.com/Ferrite-FEM/Tensors.jl/issues/233
[#236]: https://github.com/Ferrite-FEM/Tensors.jl/issues/236
[#238]: https://github.com/Ferrite-FEM/Tensors.jl/issues/238
[#240]: https://github.com/Ferrite-FEM/Tensors.jl/issues/240
[#179]: https://github.com/Ferrite-FEM/Tensors.jl/issues/179
[#197]: https://github.com/Ferrite-FEM/Tensors.jl/issues/197
[#208]: https://github.com/Ferrite-FEM/Tensors.jl/issues/208
[#223]: https://github.com/Ferrite-FEM/Tensors.jl/issues/223
[#227]: https://github.com/Ferrite-FEM/Tensors.jl/issues/227
[#239]: https://github.com/Ferrite-FEM/Tensors.jl/issues/239
[#245]: https://github.com/Ferrite-FEM/Tensors.jl/issues/245
