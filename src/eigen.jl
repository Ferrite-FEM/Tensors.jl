# MIT License: Copyright (c) 2016: Andy Ferris.
# See LICENSE.md for further licensing test

@inline function Base.eigfact{T}(S::SymmetricTensor{2, 1, T})
    @inboundsret Eigen(Vec{1, T}((S[1, 1],)), one(Tensor{2, 1, T}))
end

function Base.eigfact{T}(S::SymmetricTensor{2, 2, T})
    @inbounds begin
        # eigenvalues from quadratic formula
        trS_half = trace(S) / 2
        tmp2 = trS_half * trS_half - det(S)
        tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
        λ = Vec{2}((trS_half - tmp, trS_half + tmp))

        if S[2,1] == 0 # diagonal tensor
            Φ = one(Tensor{2, 2, T})
        else
            Φ11 = λ[1] - S[2,2]
            n1 = sqrt(Φ11 * Φ11 + S[2,1] * S[2,1])
            Φ11 = Φ11 / n1
            Φ12 = S[2,1] / n1

            Φ21 = λ[2] - S[2,2]
            n2 = sqrt(Φ21 * Φ21 + S[2,1] * S[2,1])
            Φ21 = Φ21 / n2
            Φ22 = S[2,1] / n2

            Φ = Tensor{2, 2}((Φ11, Φ12,
                              Φ21, Φ22))
        end
        return Eigen(λ, Φ)
    end
end

# A small part of the code in the following method was inspired by works of David
# Eberly, Geometric Tools LLC, in code released under the Boost Software
# License. See LICENSE.md
function Base.eigfact{T}(S::SymmetricTensor{2, 3, T})
    @inbounds begin
        R = typeof((one(T)*zero(T) + zero(T))/one(T))
        SR = convert(SymmetricTensor{2, 3, R}, S)

        S11 = SR[1, 1]; S22 = SR[2, 2]; S33 = SR[3, 3]
        S12 = SR[1, 2]; S13 = SR[1, 3]; S23 = SR[2, 3]

        p1 = abs2(S12) + abs2(S13) + abs2(S23)
        if (p1 == 0) # diagonal tensor
            v1, v2, v3 = basevec(Vec{3, R})
            if S11 < S22
                if S22 < S33
                    return Eigen(Vec{3, R}((S11, S22, S33)), Tensor{2, 3, R}((v1[1], v1[2], v1[3], v2[1], v2[2], v2[3], v3[1], v3[2], v3[3])))
                elseif S33 < S11
                    return Eigen(Vec{3, R}((S33, S11, S22)), Tensor{2, 3, R}((v3[1], v3[2], v3[3], v2[1], v2[2], v2[3], v1[1], v1[2], v1[3])))
                else
                    return Eigen(Vec{3, R}((S11, S33, S22)), Tensor{2, 3, R}((v1[1], v1[2], v1[3], v3[1], v3[2], v3[3], v2[1], v2[2], v2[3])))
                end
            else #S22 < S11
                if S11 < S33
                    return Eigen(Vec{3, R}((S22, S11, S33)), Tensor{2, 3, R}((v2[1], v2[2], v2[3], v1[1], v1[2], v1[3], v3[1], v3[2], v3[3])))
                elseif S33 < S22
                    return Eigen(Vec{3, R}((S33, S22, S11)), Tensor{2, 3, R}((v3[1], v3[2], v3[3], v2[1], v2[2], v2[3], v1[1], v1[2], v1[3])))
                else
                    return Eigen(Vec{3, R}((S22, S33, S11)), Tensor{2, 3, R}((v2[1], v2[2], v2[3], v3[1], v3[2], v3[3], v1[1], v1[2], v1[3])))
                end
            end
        end

        q = (S11 + S22 + S33) / 3
        p2 = abs2(S11 - q) + abs2(S22 - q) + abs2(S33 - q) + 2 * p1
        p = sqrt(p2 / 6)
        invp = inv(p)
        b11 = (S11 - q) * invp
        b22 = (S22 - q) * invp
        b33 = (S33 - q) * invp
        b12 = S12 * invp
        b13 = S13 * invp
        b23 = S23 * invp
        B = SymmetricTensor{2, 3, R}((b11, b12, b13, b22, b23, b33))
        r = det(B) / 2

        # In exact arithmetic for a symmetric matrix -1 <= r <= 1
        # but computation error can leave it slightly outside this range.
        if (r <= -1)
            phi = R(pi) / 3
        elseif (r >= 1)
            phi = zero(R)
        else
            phi = acos(r) / 3
        end

        λ3 = q + 2 * p * cos(phi)
        λ1 = q + 2 * p * cos(phi + (2*R(pi)/3))
        λ2 = 3 * q - λ1 - λ3 # since trace(S) = λ1 + λ2 + λ3

        if r > 0 # Helps with conditioning the eigenvector calculation
            (λ1, λ3) = (λ3, λ1)
        end

        # Calculate the first eigenvector
        # This should be orthogonal to these three rows of A - λ1*I
        # Use all combinations of cross products and choose the "best" one
        r₁ = Vec{3, R}((S11 - λ1, S12, S13))
        r₂ = Vec{3, R}((S12, S22 - λ1, S23))
        r₃ = Vec{3, R}((S13, S23, S33 - λ1))
        n₁ = r₁ ⋅ r₁
        n₂ = r₂ ⋅ r₂
        n₃ = r₃ ⋅ r₃

        r₁₂ = r₁ × r₂
        r₂₃ = r₂ × r₃
        r₃₁ = r₃ × r₁
        n₁₂ = r₁₂ ⋅ r₁₂
        n₂₃ = r₂₃ ⋅ r₂₃
        n₃₁ = r₃₁ ⋅ r₃₁

        # we want best angle so we put all norms on same footing
        # (cheaper to multiply by third nᵢ rather than divide by the two involved)
        if n₁₂ * n₃ > n₂₃ * n₁
            if n₁₂ * n₃ > n₃₁ * n₂
                Φ1 = r₁₂ / sqrt(n₁₂)
            else
                Φ1 = r₃₁ / sqrt(n₃₁)
            end
        else
            if n₂₃ * n₁ > n₃₁ * n₂
                Φ1 = r₂₃ / sqrt(n₂₃)
            else
                Φ1 = r₃₁ / sqrt(n₃₁)
            end
        end

        # Calculate the second eigenvector
        # This should be orthogonal to the previous eigenvector and the three
        # rows of A - λ2*I. However, we need to "solve" the remaining 2x2 subspace
        # problem in case the cross products are identically or nearly zero

        # The remaing 2x2 subspace is:
        if abs(Φ1[1]) < abs(Φ1[2]) # safe to set one component to zero, depending on this
            orthogonal1 = Vec{3, R}((-Φ1[3], zero(R), Φ1[1])) / sqrt(abs2(Φ1[1]) + abs2(Φ1[3]))
        else
            orthogonal1 = Vec{3, R}((zero(R), Φ1[3], -Φ1[2])) / sqrt(abs2(Φ1[2]) + abs2(Φ1[3]))
        end
        orthogonal2 = Φ1 × orthogonal1

        # The projected 2x2 eigenvalue problem is C x = 0 where C is the projection
        # of (A - λ2*I) onto the subspace {orthogonal1, orthogonal2}
        a_orth1_1 = S11 * orthogonal1[1] + S12 * orthogonal1[2] + S13 * orthogonal1[3]
        a_orth1_2 = S12 * orthogonal1[1] + S22 * orthogonal1[2] + S23 * orthogonal1[3]
        a_orth1_3 = S13 * orthogonal1[1] + S23 * orthogonal1[2] + S33 * orthogonal1[3]

        a_orth2_1 = S11 * orthogonal2[1] + S12 * orthogonal2[2] + S13 * orthogonal2[3]
        a_orth2_2 = S12 * orthogonal2[1] + S22 * orthogonal2[2] + S23 * orthogonal2[3]
        a_orth2_3 = S13 * orthogonal2[1] + S23 * orthogonal2[2] + S33 * orthogonal2[3]

        c11 = orthogonal1[1]*a_orth1_1 + orthogonal1[2]*a_orth1_2 + orthogonal1[3]*a_orth1_3 - λ2
        c12 = orthogonal1[1]*a_orth2_1 + orthogonal1[2]*a_orth2_2 + orthogonal1[3]*a_orth2_3
        c22 = orthogonal2[1]*a_orth2_1 + orthogonal2[2]*a_orth2_2 + orthogonal2[3]*a_orth2_3 - λ2

        # Solve this robustly (some values might be small or zero)
        c11² = abs2(c11)
        c12² = abs2(c12)
        c22² = abs2(c22)
        if c11² >= c22²
            if c11² > 0 || c12² > 0
                if c11² >= c12²
                    tmp = c12 / c11
                    p2 = inv(sqrt(1 + abs2(tmp)))
                    p1 = tmp * p2
                else
                    tmp = c11 / c12 # TODO check for compex input
                    p1 = inv(sqrt(1 + abs2(tmp)))
                    p2 = tmp * p1
                end
                Φ2 = p1*orthogonal1 - p2*orthogonal2
            else # c11 == 0 && c12 == 0 && c22 == 0 (smaller than c11)
                Φ2 = orthogonal1
            end
        else
            if c22² >= c12²
                tmp = c12 / c22
                p1 = inv(sqrt(1 + abs2(tmp)))
                p2 = tmp * p1
            else
                tmp = c22 / c12
                p2 = inv(sqrt(1 + abs2(tmp)))
                p1 = tmp * p2
            end
            Φ2 = p1*orthogonal1 - p2*orthogonal2
        end

        # The third eigenvector is a simple cross product of the other two
        Φ3 = Φ1 × Φ2 # should be normalized already

        # Sort them back to the original ordering, if necessary
        if r > 0
            (λ1, λ3) = (λ3, λ1)
            (Φ1, Φ3) = (Φ3, Φ1)
        end

        λ = Vec{3}((λ1, λ2, λ3))
        Φ = Tensor{2, 3}((Φ1[1], Φ1[2], Φ1[3],
                          Φ2[1], Φ2[2], Φ2[3],
                          Φ3[1], Φ3[2], Φ3[3]))
        return Eigen(λ, Φ)
    end
end
