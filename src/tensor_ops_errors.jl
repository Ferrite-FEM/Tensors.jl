# Give error for use of `*` as infix operator between tensors

# Remove `*` as infix operator between tensors
function Base.:*(S1::AbstractTensor, S2::AbstractTensor)
    error("use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction instead of `*`")
end

# Remove `'*` as infix operator between tensors
function Base.Ac_mul_B(S1::AbstractTensor, S2::AbstractTensor)
    error("use `tdot(A,B)` (or `A'⋅B`) instead of `A'*B`")
end

# Remove `.'*` as infix operator between tensors
function Base.At_mul_B(S1::AbstractTensor, S2::AbstractTensor)
    error("use `tdot(A,B)` (or `A.'⋅B`) instead if A.'*B")
end

# Remove `\` as infix operator between tensors
function Base.:\(S1::AbstractTensor, S2::AbstractTensor)
    error("use `inv(A) ⋅ B` instead of A\\B")
end

# Remove + and - between number and Tensor (issue #75)
Base.:+(n::Number, T::AbstractTensor) = throw(MethodError(+, (n, T)))
Base.:+(T::AbstractTensor, n::Number) = throw(MethodError(+, (T, n)))
Base.:-(n::Number, T::AbstractTensor) = throw(MethodError(-, (n, T)))
Base.:-(T::AbstractTensor, n::Number) = throw(MethodError(-, (T, n)))
