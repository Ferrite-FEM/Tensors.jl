# ContMechTensors

*Efficient computations with symmetric and unsymmetric tensors in Julia.*

| **Documentation**                                                               | **Build Status**                                                |
|:-------------------------------------------------------------------------------:|:---------------------------------------------------------------:|
| [![][docs-stable-img]][docs-stable-url] [![][docs-latest-img]][docs-latest-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] |

## Introduction

This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4.
The Tensors are allocated on the stack which means that there is no need to preallocate output results for performance. 
Unicode infix operators are provided such that the tensor expression in the source code is similar to the one written with mathematical notation.
When possible, symmetry of tensors is exploited for better performance.

## Installation

The package is registered in `METADATA.jl` and so can be installed with `Pkg.add`.

```julia
julia> Pkg.add("ContMechTensors")
```

## Documentation

- [**STABLE**][docs-stable-url] &mdash; **most recently tagged version of the documentation.**
- [**LATEST**][docs-latest-url] &mdash; *in-development version of the documentation.*

## Project Status

The package is tested against Julia `0.5`, and `0.6-dev` on Linux, OS X, and Windows.

## Contributing and Questions

Contributions are very welcome, as are feature requests and suggestions. Please open an [issue][issues-url] if you encounter any problems.

[docs-latest-img]: https://img.shields.io/badge/docs-latest-blue.svg
[docs-latest-url]: https://kristofferc.github.io/ContMechTensors.jl/latest/

[docs-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[docs-stable-url]: https://kristofferc.github.io/ContMechTensors.jl/stable

[travis-img]: https://travis-ci.org/KristofferC/ContMechTensors.jl.svg?branch=master
[travis-url]: https://travis-ci.org/KristofferC/ContMechTensors.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/xe0ghtyas12wv555/branch/master?svg=true
[appveyor-url]: https://ci.appveyor.com/project/KristofferC/contmechtensors-jl/branch/master

[issues-url]: https://github.com/KristofferC/ContMechTensors.jl/issues
