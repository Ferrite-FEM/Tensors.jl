# MIT License: Copyright (c) 2016: Andy Ferris.
# See LICENSE.md for further licensing test

@inline function LinearAlgebra.eigen(S::SymmetricTensor{2, 1, T}) where {T}
    @inbounds Eigen(Vec{1, T}((S[1, 1],)), one(Tensor{2, 1, T}))
end

function LinearAlgebra.eigen(R::SymmetricTensor{2, 2, T′}) where T′
    S = ustrip(R)
    T = eltype(S)
    @inbounds begin
        if S[2,1] == 0 # diagonal tensor
            S11 = S[1,1]
            S22 = S[2,2]
            if S11 < S22
                λ = Vec{2}((S11, S22))
                Φ = Tensor{2,2,T}((T(1), T(0), T(0), T(1)))
            else # S22 <= S11
                λ = Vec{2}((S22, S11))
                Φ = Tensor{2,2,T}((T(0), T(1), T(1), T(0)))
            end
        else
            # eigenvalues from quadratic formula
            trS_half = tr(S) / 2
            tmp2 = trS_half * trS_half - det(S)
            tmp2 < 0 ? tmp = zero(tmp2) : tmp = sqrt(tmp2) # Numerically stable for identity matrices, etc.
            λ = Vec{2}((trS_half - tmp, trS_half + tmp))

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
        return Eigen(convert(Vec{2,T′}, λ), Φ)
    end
end

# Port of https://www.geometrictools.com/GTEngine/Include/Mathematics/GteSymmetricEigensolver3x3.h
# released by David Eberly, Geometric Tools, Redmond WA 98052
# under the Boost Software License, Version 1.0 (included at the end of this file)
# The original documentation states
# (see https://www.geometrictools.com/Documentation/RobustEigenSymmetric3x3.pdf )
# [This] is an implementation of Algorithm 8.2.3 (Symmetric QR Algorithm) described in
# Matrix Computations,2nd edition, by G. H. Golub and C. F. Van Loan, The Johns Hopkins
# University Press, Baltimore MD, Fourth Printing 1993. Algorithm 8.2.1 (Householder
# Tridiagonalization) is used to reduce matrix A to tridiagonal D′. Algorithm 8.2.2
# (Implicit Symmetric QR Step with Wilkinson Shift) is used for the iterative reduction
# from tridiagonal to diagonal. Numerically, we have errors E=RTAR−D. Algorithm 8.2.3
# mentions that one expects |E| is approximately μ|A|, where |M| denotes the Frobenius norm
# of M and where μ is the unit roundoff for the floating-point arithmetic: 2−23 for float,
# which is FLTEPSILON = 1.192092896e-7f, and 2−52 for double, which is
# DBLEPSILON = 2.2204460492503131e-16.
# TODO ensure right-handedness of the eigenvalue matrix
function LinearAlgebra.eigen(R::SymmetricTensor{2,3,T′}) where T′
    A = ustrip(R)
    T = eltype(A)
    function converged(aggressive, bdiag0, bdiag1, bsuper)
        if aggressive
            bsuper == 0
        else
            diag_sum = abs(bdiag0) + abs(bdiag1)
            diag_sum + bsuper == diag_sum
        end
    end

    function get_cos_sin(u::T,v::T) where {T}
        max_abs = max(abs(u), abs(v))
        if max_abs > 0
            u,v = (u,v) ./ max_abs
            len = sqrt(u^2 + v^2)
            cs, sn = (u,v) ./ len
            if cs > 0
                cs = -cs
                sn = -sn
            end
            T(cs), T(sn)
        else
            T(-1), T(0)
        end
    end

    function _sortperm3(v)
        local perm = (1,2,3)
        # unrolled bubble-sort
        (v[perm[1]] > v[perm[2]]) && (perm = (perm[2], perm[1], perm[3]))
        (v[perm[2]] > v[perm[3]]) && (perm = (perm[1], perm[3], perm[2]))
        (v[perm[1]] > v[perm[2]]) && (perm = (perm[2], perm[1], perm[3]))
        perm
    end

    # Givens reflections
    update0(Q, c::T, s::T) where T = Q ⋅ Tensor{2,3}((c, s, T(0), T(0), T(0), T(1), -s, c, T(0)))
    update1(Q, c::T, s::T) where T = Q ⋅ Tensor{2,3}((T(0), c, -s, T(1), T(0), T(0), T(0), s, c))
    # Householder reflections
    update2(Q, c::T, s::T) where T = Q ⋅ Tensor{2,3}((c, s, T(0), s, -c, T(0), T(0), T(0), T(1)))
    update3(Q, c::T, s::T) where T = Q ⋅ Tensor{2,3}((T(1), T(0), T(0), T(0), c, s, T(0), s, -c))

    is_rotation = false

    # If `aggressive` is `true`, the iterations occur until a superdiagonal
    # entry is exactly zero, otherwise they occur until it is effectively zero
    # compared to the magnitude of its diagonal neighbors. Generally the non-
    # aggressive convergence is acceptable.
    #
    # Even with `aggressive = true` this method is faster than the one it
    # replaces and in order to keep the old interface, aggressive is set to true
    aggressive = true

    # the input is symmetric, so we only consider the unique elements:
    a00, a01, a02, a11, a12, a22 = A[1,1], A[1,2], A[1,3], A[2,2], A[2,3], A[3,3]

    # Compute the Householder reflection H and B = H * A * H where b02 = 0

    c, s = get_cos_sin(a12, -a02)

    Q = Tensor{2,3}((c, s, T(0), s, -c, T(0), T(0), T(0), T(1)))

    term0 = c * a00 + s * a01
    term1 = c * a01 + s * a11
    b00 = c * term0 + s * term1
    b01 = s * term0 - c * term1
    term0 = s * a00 - c * a01
    term1 = s * a01 - c * a11
    b11 = s * term0 - c * term1
    b12 = s * a02 - c * a12
    b22 = a22

    # Givens reflections, B' = G^T * B * G, preserve tridiagonal matrices
    max_iteration = 2 * (1 + precision(T) - exponent(floatmin(T)))

    if abs(b12) <= abs(b01)
        saveB00, saveB01, saveB11 = b00, b01, b11
        for _ in 1:max_iteration
            # compute the Givens reflection
            c2, s2 = get_cos_sin((b00 - b11) / 2, b01)
            s = sqrt((1 - c2) / 2)
            c = s2 / 2s

            # update Q by the Givens reflection
            Q = update0(Q, c, s)
            is_rotation = !is_rotation

            # update B ← Q^T * B * Q, ensuring that b02 is zero and |b12| has
            # strictly decreased
            saveB00, saveB01, saveB11 = b00, b01, b11
            term0 = c * saveB00 + s * saveB01
            term1 = c * saveB01 + s * saveB11
            b00 = c * term0 + s * term1
            b11 = b22
            term0 = c * saveB01 - s * saveB00
            term1 = c * saveB11 - s * saveB01
            b22 = c * term1 - s * term0
            b01 = s * b12
            b12 = c * b12

            if converged(aggressive, b00, b11, b01)
                # compute the Householder reflection
                c2, s2 = get_cos_sin((b00 - b11) / 2, b01)
                s = sqrt((1 - c2) / 2)
                c = s2 / 2s

                # update Q by the Householder reflection
                Q = update2(Q, c, s)
                is_rotation = !is_rotation

                # update D = Q^T * B * Q
                saveB00, saveB01, saveB11 = b00, b01, b11
                term0 = c * saveB00 + s * saveB01
                term1 = c * saveB01 + s * saveB11
                b00 = c * term0 + s * term1
                term0 = s * saveB00 - c * saveB01
                term1 = s * saveB01 - c * saveB11
                b11 = s * term0 - c * term1
                break
            end
        end
    else
        saveB11, saveB12, saveB22 = b11, b12, b22
        for _ in 1:max_iteration
            # compute the Givens reflection
            c2, s2 = get_cos_sin((b22 - b11) / 2, b12)
            s = sqrt((1 - c2) / 2)
            c = s2 / 2s

            # update Q by the Givens reflection
            Q = update1(Q, c, s)
            is_rotation = !is_rotation

            # update B ← Q^T * B * Q ensuring that b02 is zero and |b12| has
            # strictly decreased.
            saveB11, saveB12, saveB22 = b11, b12, b22

            term0 = c * saveB22 + s * saveB12
            term1 = c * saveB12 + s * saveB11
            b22 = c * term0 + s * term1
            b11 = b00
            term0 = c * saveB12 - s * saveB22
            term1 = c * saveB11 - s * saveB12
            b00 = c * term1 - s * term0
            b12 = s * b01
            b01 = c * b01

            if converged(aggressive, b11, b22, b12)
                # compute the Householder reflection
                c2, s2 = get_cos_sin((b11 - b22) / 2, b12)
                s = sqrt((1 - c2) / 2)
                c = s2 / 2s

                # update Q by the Householder reflection
                Q = update3(Q, c, s)
                is_rotation = !is_rotation

                # update D = Q^T * B * Q
                saveB11, saveB12, saveB22 = b11, b12, b22
                term0 = c * saveB11 + s * saveB12
                term1 = c * saveB12 + s * saveB22
                b11 = c * term0 + s * term1
                term0 = s * saveB11 - c * saveB12
                term1 = s * saveB12 - c * saveB22
                b22 = s * term0 - c * term1
                break
            end
        end
    end
    evals = convert(Vec{3,T′}, Vec{3}((b00, b11, b22)))
    perm = _sortperm3(evals)
    evals_sorted = Vec{3}((evals[perm[1]], evals[perm[2]], evals[perm[3]]))
    Q_sorted = Tensor{2,3}((
        Q[1, perm[1]], Q[2, perm[1]], Q[3, perm[1]],
        Q[1, perm[2]], Q[2, perm[2]], Q[3, perm[2]],
        Q[1, perm[3]], Q[2, perm[3]], Q[3, perm[3]],
    ))
    return Eigen(evals_sorted, Q_sorted)
end
