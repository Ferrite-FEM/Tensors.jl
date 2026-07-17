# specialized methods
"""
    dotdot(::Vec, ::FourthOrderTensor, ::Vec)

Computes a special dot product between two vectors and a fourth order tensor
such that ``a_k C_{ikjl} b_l``.
"""
@tensorop function dotdot(v1::AbstractTensor{1}, S::FourthOrderTensor, v2::AbstractTensor{1})
    @muladd C[i, j] = v1[k] * S[i, k, j, l] * v2[l]
end
