# Remove `*` as infix operator between tensors
function Base.:*(S1::AbstractTensor, S2::AbstractTensor)
    error("use `⋅` (`\\cdot`) for single contraction and `⊡` (`\\boxdot`) for double contraction instead of `*`")
end

for f in (:transpose, :adjoint)
    @eval function LinearAlgebra.$f(::Vec)
        throw(ArgumentError("the (no-op) $($f) is discontinued for `Tensors.Vec`"))
    end
end
