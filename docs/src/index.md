# ContMechTensors

*Efficient computations with symmetric and non-symmetric tensors in Julia.*

## Introduction

This Julia package provides fast operations with symmetric and non-symmetric tensors of order 1, 2 and 4 in 1, 2 and 3 dimensions.
The tensors are stack allocated which means that there is no need to preallocate results of operations and nice infix notation can be used without a performance penalty.
For the symmetric tensors, when possible, the symmetry is exploited for better performance.

## Installation

`ContMechTensors` is a registered package and so can be installed via

```julia
Pkg.add("ContMechTensors")
```

The package has no dependencies other than Julia (`0.5` and up) itself.

## Manual Outline

```@contents
Pages = [
    "man/constructing_tensors.md",
    "man/indexing.md",
    "man/binary_operators.md",
    "man/other_operators.md",
    "man/storing_tensors.md",
]
Depth = 1
```

## Demos

```@contents
Pages = [
    "demos.md"]
Depth = 1
```




