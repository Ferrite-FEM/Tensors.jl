# Other operators

For vectors (first order tensors): `norm`

For second order tensors: `norm`, `trace` (`vol`), `det`, `inv`, `transpose`, `symmetric`, `skew`, `eig`, `mean` defined as `trace(s) / 3`, and `dev` defined as `s - mean(s) * I`.

For fourth order tensors: `norm`, `trace`, `symmetric` (same as `minorsymmetric`), `majorsymmetric`, `transpose` (same as `minortranspose`), `majortranspose`, `permute_index`

There is also a few special functions that can be convenient:

* For computing `F' ⋅ F` between two general second order tensors there is `tdot(F)` which returns a `SymmetricTensor`.

* For computing $a_k \cdot \mathbf{C}_{ikjl} \cdot b_l$ for two vectors $a$ and $b$ and a fourth order symmetric tensor $\mathbf{C}$ there is `dotdot(a, C, b)`. This function is useful because it is the expression for the tangent matrix in continuum mechanics when the displacements are approximated by scalar base functions.
