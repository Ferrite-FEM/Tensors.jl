# Give error for use of `*` as infix operator between tensors

# Remove `*` as infix operator between tensors
function Base.:*(S1::AbstractTensor, S2::AbstractTensor)
    error("Don't use `*` for multiplication between tensors. Use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction.")
end

# Remove `'*` as infix operator between tensors
function Base.Ac_mul_B(S1::AbstractTensor, S2::AbstractTensor)
    error("Don't use `A'*B`, use `tdot(A,B)` (or `A'⋅B`) instead.")
end

# Remove `.'*` as infix operator between tensors
function Base.At_mul_B(S1::AbstractTensor, S2::AbstractTensor)
    error("Don't use `A.'*B`, use `tdot(A,B)` (or `A.'⋅B`) instead.")
end

# Remove `\` as infix operator between tensors
function Base.:\(S1::AbstractTensor, S2::AbstractTensor)
    error("Don't use `A\\B`, use `inv(A) ⋅ B` instead.")
end

# Remove + and - between number and Tensor (issue #75)
Base.:+(n::Number, T::AbstractTensor) = throw(MethodError(+, (n, T)))
Base.:+(T::AbstractTensor, n::Number) = throw(MethodError(+, (T, n)))
Base.:-(n::Number, T::AbstractTensor) = throw(MethodError(-, (n, T)))
Base.:-(T::AbstractTensor, n::Number) = throw(MethodError(-, (T, n)))
