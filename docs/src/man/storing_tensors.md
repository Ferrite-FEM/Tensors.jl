# Storing tensors

Even though a user mostly deals with the `Tensor{order, dim, T}` parameters, the full parameter list for a tensor is actually `Tensor{order, dim, T, N}` where `N` is the number of independent elements in the tensor. The reason for this is that the internal storage is a `NTuple{N, T}`. In order to get good performance when storing tensors in other types it is importatant that the container type is also parametrized on `N`. For example, when storing one symmetric second order tensor and one unsymmetric tensor, this is the preferred way:

```julia
immutable Container{dim, T, N, M}
    sym_tens::SymmetricTensor{2, dim, T, N}
    tens::Tensor{2, dim, T, M}
end
```

Leaving out the `M` and `N` would lead to bad performance.

!!! tip
    The number of independent elements `N` are already included in the `typealias` `Vec` so they can be stored with e.g.
    ```julia
    immutable VecContainer{dim, T}
        vec::Vec{dim, T}
    end
    ```
    without giving bad performance.
