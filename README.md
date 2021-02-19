# Tensors.jl

*Efficient computations with symmetric and non-symmetric tensors with support for automatic differentiation.*

| **Documentation**                                                               | **Build Status**                                                                                |
|:-------------------------------------------------------------------------------:|:-----------------------------------------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] | [![][ci-img]][ci-url] [![][codecov-img]][codecov-url] |

## Introduction

This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4.
The tensors are allocated on the stack which means that there is no need to preallocate output results for performance.
Unicode infix operators are provided such that the tensor expression in the source code is similar to the one written with mathematical notation.
When possible, symmetry of tensors is exploited for better performance.
Supports Automatic Differentiation to easily compute first and second order derivatives of tensorial functions.

## Installation

The package can be installed with the Julia package manager.
From the Julia REPL, type `]` to enter the Pkg REPL mode and run:

```
pkg> add Tensors
```

Or, equivalently, via the `Pkg` API:

```julia
julia> import Pkg; Pkg.add("Tensors")
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.**
- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

## Project Status

The package is tested against Julia `1.X` on Linux, macOS, and Windows.

## Contributing and Questions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

### Things to work on

If you are interested in contributing to Tensors.jl, here are a few topics that can get you started:

* Implement support for third order tensors. These are more rarely used than first, second and fourth order tensors but are still useful in some applications. It would be good to support this.
* Find a way to reduce code duplication without sacrificing performance or compilation time. Currently, there is quite a lot of code duplication in the implementation of different operators. It should be possible to have a higher level code generation framework that generates optimized functions from pretty much only the Einstein summation notation for the operation.
* Tensors.jl has been developed with mostly the application to continuum mechanics in mind. For other fields, perhaps other tensor operations are useful. Implement these in a well performant manner and give good test coverage and documentation for the new functionalities.

## Citing Tensors.jl

If you use Tensors.jl for research and publication, please cite the following article
```
@article{Tensors.jl,
  title = {Tensors.jl -- Tensor Computations in Julia},
  author = {Carlsson, Kristoffer and Ekre, Fredrik},
  year = {2019},
  journal = {Journal of Open Research Software},
  doi = {10.5334/jors.182},
}
```

## Related packages

Both the packages below provide a convenience macro to provide einstein summation notation for standard Julia `Array`'s:

* [Einsum.jl](https://github.com/ahwillia/Einsum.jl)
* [TensorOperations.jl](https://github.com/Jutho/TensorOperations.jl)


[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://kristofferc.github.io/Tensors.jl/stable

[ci-img]: https://github.com/Ferrite-FEM/Tensors.jl/workflows/CI/badge.svg?branch=master
[ci-url]: https://github.com/Ferrite-FEM/Tensors.jl/actions?query=workflows%3ACI+branch%3Amaster

[issues-url]: https://github.com/Ferrite-FEM/Tensors.jl/issues

[codecov-img]: https://codecov.io/gh/Ferrite-FEM/Tensors.jl/branch/master/graph/badge.svg?branch=master
[codecov-url]: https://codecov.io/gh/Ferrite-FEM/Tensors.jl?branch=master
