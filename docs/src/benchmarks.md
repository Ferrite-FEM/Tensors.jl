# Benchmarks

Here are some benchmark timings for tensors in 3 dimensions. For comparison
the timings for the same operations using standard Julia `Array`s are also
presented.

In the table below, `a` denotes a vector, `A`, `As` denotes second order
non-symmetric and symmetric tensors and `AA`, `AAs` denotes fourth order
non-symmetric and symmetric tensors respectively.

| Operation  | `Tensor` | `Array` | speed-up |
|:-----------|---------:|--------:|---------:|
| **Single contraction** | | | |
| `a ⋅ a` | 1.241 ns | 9.795 ns | ×7.9 |
| `A ⋅ a` | 2.161 ns | 58.769 ns | ×27.2 |
| `A ⋅ A` | 3.117 ns | 44.395 ns | ×14.2 |
| `As ⋅ As` | 5.125 ns | 44.498 ns | ×8.7 |
| **Double contraction** | | | |
| `A ⊡ A` | 1.927 ns | 12.189 ns | ×6.3 |
| `As ⊡ As` | 1.927 ns | 12.187 ns | ×6.3 |
| `AA ⊡ A` | 6.087 ns | 78.554 ns | ×12.9 |
| `AA ⊡ AA` | 60.820 ns | 280.502 ns | ×4.6 |
| `AAs ⊡ AAs` | 22.104 ns | 281.003 ns | ×12.7 |
| `As ⊡ AAs ⊡ As` | 9.466 ns | 89.747 ns | ×9.5 |
| **Outer product** | | | |
| `a ⊗ a` | 2.167 ns | 32.447 ns | ×15.0 |
| `A ⊗ A` | 9.801 ns | 86.568 ns | ×8.8 |
| `As ⊗ As` | 4.311 ns | 87.830 ns | ×20.4 |
| **Other operations** | | | |
| `det(A)` | 1.924 ns | 177.134 ns | ×92.1 |
| `det(As)` | 1.924 ns | 182.831 ns | ×95.0 |
| `inv(A)` | 6.099 ns | 595.591 ns | ×97.7 |
| `inv(As)` | 4.587 ns | 635.858 ns | ×138.6 |
| `norm(a)` | 1.494 ns | 9.838 ns | ×6.6 |
| `norm(A)` | 1.990 ns | 16.752 ns | ×8.4 |
| `norm(As)` | 2.011 ns | 16.757 ns | ×8.3 |
| `norm(AA)` | 9.283 ns | 28.125 ns | ×3.0 |
| `norm(AAs)` | 5.422 ns | 28.134 ns | ×5.2 |
| `a × a` | 1.921 ns | 32.736 ns | ×17.0 |


The benchmarks are generated by
[`benchmark_doc.jl`](https://github.com/Ferrite-FEM/Tensors.jl/blob/master/benchmark/benchmark_doc.jl)
on the following system:

```
julia> versioninfo()

Julia Version 0.6.0-pre.beta.297
Commit 2a61131* (2017-04-24 23:57 UTC)
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: Intel(R) Core(TM) i5-7600K CPU @ 3.80GHz
  WORD_SIZE: 64
  BLAS: libopenblas (USE64BITINT NO_AFFINITY HASWELL)
  LAPACK: libopenblas64_
  LIBM: libopenlibm
  LLVM: libLLVM-3.9.1 (ORCJIT, broadwell)
```
