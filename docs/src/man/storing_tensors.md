# Storing tensors

The full parameter list for a tensor is `Tensor{order, dim, T}`. The number of independent elements stored in the internal `NTuple` is computed directly from `order` and `dim`, so `Tensor{order, dim, T}` and `SymmetricTensor{order, dim, T}` are concrete types. Storing tensors in other types is therefore as simple as:

```julia
struct Container{dim, T}
    sym_tens::SymmetricTensor{2, dim, T}
    tens::Tensor{2, dim, T}
end
```

This gives optimal performance without any extra type parameters.
