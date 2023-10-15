# Tensors changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## Unreleased

## [1.15.1]
### Added
 - Partial support for 3rd order Tensors [#205][github-205]
    * All construction methods, e.g. `zero(Tensor{3})`, `rand(Tensor{3})`, `Tensor{3}((i,j,k)->f(i,j,k))`
    * Gradient of 2nd order tensor wrt. vector
    * `rotate(::Tensor{3})`
    * `dcontract(::Tensor{D1}, ::Tensor{D2})` for (D1,D2) in ((2,3), (3,2), (3,4), (4,3))
    * `otimes(::Vec, ::SecondOrderTensor)` and `otimes(::SecondOrderTensor, ::Vec)`
    * `dot(::Tensor{D1}, ::Tensor{D2})` for (D1,D2) in ((3,1), (1,3), (2,3), (3,2))

<!-- Release links -->
[Unreleased]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v1.15.1...HEAD
[1.15.1]: https://github.com/Ferrite-FEM/Ferrite.jl/compare/v1.15.0...v1.15.1

<!-- GitHub pull request/issue links -->
[github-205]: https://github.com/Ferrite-FEM/Tensors.jl/pull/205