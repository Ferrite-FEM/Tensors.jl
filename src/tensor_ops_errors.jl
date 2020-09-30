# Remove `*` as infix operator between tensors
function Base.:*(S1::AbstractTensor, S2::AbstractTensor)
    error("use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction instead of `*`")
end