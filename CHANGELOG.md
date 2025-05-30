# Tensors changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
[#205]: https://github.com/Ferrite-FEM/Tensors.jl/issues/205
[#212]: https://github.com/Ferrite-FEM/Tensors.jl/issues/212
[#225]: https://github.com/Ferrite-FEM/Tensors.jl/issues/225
