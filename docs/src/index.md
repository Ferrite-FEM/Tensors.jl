# Tensors

*Efficient computations with symmetric and non-symmetric tensors in Julia.*

## Introduction

This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4.
The Tensors are allocated on the stack which means that there is no need to preallocate output results for performance.
Unicode infix operators are provided such that the tensor expression in the source code is similar to the one written with mathematical notation.
When possible, symmetry of tensors is exploited for better performance.
Supports Automatic Differentiation to easily compute first and second order derivatives of tensorial functions.

## Installation

`Tensors` is a registered package and so can be installed via

```julia
Pkg.add("Tensors")
```

## Manual Outline

```@contents
Pages = [
    "man/constructing_tensors.md",
    "man/indexing.md",
    "man/binary_operators.md",
    "man/other_operators.md",
    "man/storing_tensors.md",
    "man/automatic_differentiation.md",
]
Depth = 1
```

## Benchmarks

```@contents
Pages = [
    "benchmarks.md"]
```

## Demos

```@contents
Pages = [
    "demos.md"]
Depth = 1
```
