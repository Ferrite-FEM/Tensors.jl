# Regression tests for the einsum engine (src/einsum.jl, src/simd_lowering.jl):
# plan structure, storage order, symmetric multiplicities, SIMD/scalar path
# agreement, output types, and the operations added with the engine.

using Tensors, Test, LinearAlgebra
using Statistics: mean
import ForwardDiff

# exact-arithmetic tensors: small integer values stored as T, so products and
# sums are exact and path differences (SIMD vs scalar, muladd vs mul-add)
# cannot hide structural errors (wrong index, wrong multiplicity)
exact_rand(TT) = Tensors.get_base(TT)(Tuple(rand(-3:3, Tensors.n_components(Tensors.get_base(TT)))))

# brute-force reference: contract the `nc` trailing/leading indices via Array
function ref_contract(A::AbstractTensor, B::AbstractTensor, nc::Int)
    a = Array(A); b = Array(B)
    sa = size(a); sb = size(b)
    na = length(sa) - nc
    out = zeros(promote_type(eltype(a), eltype(b)), sa[1:na]..., sb[(nc + 1):end]...)
    for ia in CartesianIndices(sa[1:na]), ib in CartesianIndices(sb[(nc + 1):end])
        s = zero(eltype(out))
        for k in CartesianIndices(sa[(na + 1):end])
            s += a[ia, k] * b[k, ib]
        end
        out[ia, ib] = s
    end
    return out
end

@testset "einsum engine" begin
    @testset "plan structure: storage order and symmetric multiplicities" begin
        # symmetric storage order is lower-triangle column-major, composed for order 4
        @test Tensors.base_components(SymmetricTensor{2, 3}) ==
              [(1, 1), (2, 1), (3, 1), (2, 2), (3, 2), (3, 3)]
        @test Tensors.symmetric_indices(2, 3) == [1, 2, 3, 5, 6, 9]
        @test Tensors.symmetric_multiplicities(2, 3) == [1, 2, 2, 1, 2, 1]
        @test sum(Tensors.symmetric_multiplicities(4, 3)) == 81
        for dim in 1:3
            full = Tensors.base_components(Tensor{2, dim})
            @test length(full) == dim^2
            # compute_index consistent with base_components enumeration
            for (n, c) in enumerate(full)
                @test Tensors.compute_index(Tensor{2, dim}, c...) == n
            end
            for (n, c) in enumerate(Tensors.base_components(SymmetricTensor{4, dim}))
                @test Tensors.compute_index(SymmetricTensor{4, dim}, c...) == n
            end
        end
    end

    @testset "contractions vs Array reference (exact arithmetic), dim $dim" for dim in (1, 2, 3)
        A1 = exact_rand(Tensor{1, dim}); A2 = exact_rand(Tensor{2, dim}); A3 = exact_rand(Tensor{3, dim})
        A4 = exact_rand(Tensor{4, dim}); S2 = exact_rand(SymmetricTensor{2, dim}); S4 = exact_rand(SymmetricTensor{4, dim})
        # single contraction
        @test Array(A2 ⋅ A1) == ref_contract(A2, A1, 1)
        @test Array(S2 ⋅ A2) == ref_contract(S2, A2, 1)
        @test Array(S4 ⋅ S2) == ref_contract(S4, S2, 1)
        # double contraction (incl. symmetric mixes)
        @test A2 ⊡ A2 == ref_contract(A2, A2, 2)[]
        @test S2 ⊡ A2 == ref_contract(S2, A2, 2)[]
        @test Array(A4 ⊡ A2) == ref_contract(A4, A2, 2)
        @test Array(S4 ⊡ A2) == ref_contract(S4, A2, 2)
        @test Array(S4 ⊡ S4) == ref_contract(S4, S4, 2)
        @test Array(S4 ⊡ A3) == ref_contract(S4, A3, 2)
        @test Array(A3 ⊡ S4) == ref_contract(A3, S4, 2)
        # operations added with the engine (previously missing combinations)
        @test Array(A3 ⋅ A3) == ref_contract(A3, A3, 1)         # issue #227
        @test Array(A1 ⋅ A4) == ref_contract(A1, A4, 1)
        @test Array(A4 ⋅ A1) == ref_contract(A4, A1, 1)
        @test Array(A1 ⊗ A3) == ref_contract(A1, A3, 0)
        @test Array(A3 ⊗ A1) == ref_contract(A3, A1, 0)
        @test (A3 ⋅ A3) isa Tensor{4, dim}
    end

    @testset "SIMD and scalar paths agree exactly (exact arithmetic)" begin
        for dim in (2, 3), T in (Float32, Float64)
            A2 = exact_rand(Tensor{2, dim, T}); S2 = exact_rand(SymmetricTensor{2, dim, T})
            A4 = exact_rand(Tensor{4, dim, T}); S4 = exact_rand(SymmetricTensor{4, dim, T})
            A3 = exact_rand(Tensor{3, dim, T})
            f64(x) = Tensors.get_base(typeof(x))(map(Float64, Tensors.get_data(x)))
            asarr(x::Number) = fill(Float64(x))
            asarr(x) = Float64.(Array(x))
            for (a, b) in ((A2, A2), (S2, S2), (S2, A2), (A4, A2), (S4, S2), (S4, A2),
                           (A4, A4), (S4, S4), (S4, A4), (S4, A3), (A2, A4), (A2, S4))
                @test asarr(dcontract(a, b)) == asarr(dcontract(f64(a), f64(b)))
            end
            for (a, b) in ((A2, A2), (S2, S2), (A2, S2))
                @test asarr(dot(a, b)) == asarr(dot(f64(a), f64(b)))
                @test asarr(otimes(a, b)) == asarr(otimes(f64(a), f64(b)))
            end
        end
    end

    @testset "output types" begin
        S2 = rand(SymmetricTensor{2, 3}); A2 = rand(Tensor{2, 3})
        S4 = rand(SymmetricTensor{4, 3})
        @test otimes(S2, S2) isa SymmetricTensor{4, 3}
        @test otimes(S2, A2) isa Tensor{4, 3}
        @test dcontract(S4, S2) isa SymmetricTensor{2, 3}
        @test dcontract(S4, A2) isa SymmetricTensor{2, 3}  # symmetry carried by S4's leading pair
        @test dcontract(A2, S4) isa SymmetricTensor{2, 3}
        @test dot(S2, S2) isa Tensor{2, 3}                 # product of symmetric is NOT symmetric
        # mixed dims collapse to Tensor when equal
        m23 = rand(MixedTensor{2, Tuple{2, 3}})
        @test (m23 ⋅ m23') isa Tensor{2, 2}
        @test (m23' ⋅ m23) isa Tensor{2, 3}
        @test (m23 ⊗ rand(Vec{2})) isa MixedTensor{3, Tuple{2, 3, 2}}
        @test transpose(rand(MixedTensor{2, Tuple{2, 2}})) isa Tensor{2, 2}
    end

    @testset "runtime dimension errors" begin
        m23 = rand(MixedTensor{2, Tuple{2, 3}})
        @test_throws DimensionMismatch m23 ⋅ m23
        @test_throws DimensionMismatch m23 ⊡ m23'
        m123 = rand(MixedTensor{3, Tuple{1, 2, 3}})
        @test_throws ArgumentError m123[:]
        @test_throws ArgumentError tr(m23)
        @test_throws ArgumentError mean(m23)
        @test_throws ArgumentError vol(m23)
        @test_throws ArgumentError dev(m23)
        @test_throws ArgumentError ismajorsymmetric(rand(MixedTensor{4, Tuple{2, 2, 2, 3}}))
    end

    @testset "three-argument operations" begin
        for dim in (1, 2, 3)
            v1, v2 = exact_rand(Vec{dim}), exact_rand(Vec{dim})
            for C in (exact_rand(SymmetricTensor{4, dim}), exact_rand(Tensor{4, dim}))
                ref = [sum(v1[k] * C[i, k, j, l] * v2[l] for k in 1:dim, l in 1:dim) for i in 1:dim, j in 1:dim]
                @test Array(dotdot(v1, C, v2)) == ref
                @test (@inferred dotdot(v1, C, v2)) isa Tensor{2, dim}
            end
        end
        # a ternary declaration through the macro: fused chained contraction
        Tensors.@tensorop function my_chain3(A::SecondOrderTensor, B::SecondOrderTensor, C::SecondOrderTensor)
            @muladd D[i, l] = A[i, j] * B[j, k] * C[k, l]
        end
        A, B, C = exact_rand(Tensor{2, 3}), exact_rand(SymmetricTensor{2, 3}), exact_rand(Tensor{2, 3})
        @test my_chain3(A, B, C) == A ⋅ B ⋅ C
        @test (@inferred my_chain3(A, B, C)) isa Tensor{2, 3}
        m = exact_rand(MixedTensor{2, Tuple{2, 3}})
        @test my_chain3(m, B, m') ≈ m ⋅ B ⋅ m'
        @test my_chain3(m, B, m') isa Tensor{2, 2}
        # scalar output
        Tensors.@tensorop function my_vSv(a::AbstractTensor{1}, S::SecondOrderTensor, b::AbstractTensor{1})
            @muladd r = a[i] * S[i, j] * b[j]
        end
        a, b = exact_rand(Vec{3}), exact_rand(Vec{3})
        @test my_vSv(a, B, b) ≈ a ⋅ (B ⋅ b)
        @test (@inferred my_vSv(a, B, b)) isa Int  # exact_rand tensors hold Ints
        # an index in three arguments is a definition-time error
        @test_throws ErrorException Tensors.einsum_expr((:i,),
            Tensors.IndexedArg(:A, Tensor{2, 3}, (:i, :j), Any),
            Tensors.IndexedArg(:B, Tensor{2, 3}, (:j, :k), Any),
            Tensors.IndexedArg(:C, Tensor{2, 3}, (:j, :k), Any))
    end

    @testset "dotdot bilinear forms" begin
        for dim in (1, 2, 3)
            a, b = rand(Vec{dim}), rand(Vec{dim})
            S2, T2 = rand(SymmetricTensor{2, dim}), rand(Tensor{2, dim})
            S4, T4 = rand(SymmetricTensor{4, dim}), rand(Tensor{4, dim})
            @test dotdot(a, S2, b) ≈ a ⋅ (S2 ⋅ b)
            @test dotdot(a, T2, b) ≈ a ⋅ (T2 ⋅ b)
            @test (@inferred dotdot(a, S2, b)) isa Float64
            @test dotdot(S2, S4, S2) ≈ S2 ⊡ S4 ⊡ S2
            @test dotdot(T2, T4, S2) ≈ T2 ⊡ T4 ⊡ S2
            @test (@inferred dotdot(S2, S4, S2)) isa Float64
        end
    end

    @testset "propagate_gradient with several active arguments" begin
        # g(F, C) with both arguments depending on the differentiated variable
        myg(F, C) = tr(F ⋅ C)
        myg_dg(F, C) = (myg(F, C), (transpose(C), transpose(F)))
        myg(F::Tensor{2, 3, <:ForwardDiff.Dual}, C) = propagate_gradient(myg_dg, Val((1, 2)), F, C)
        F0 = rand(Tensor{2, 3})
        @test gradient(F -> myg(F, F ⋅ F), F0) ≈ gradient(F -> tr(F ⋅ (F ⋅ F)), F0)
        @test hessian(F -> myg(F, F ⋅ F), F0) ≈ hessian(F -> tr(F ⋅ F ⋅ F), F0)
        # an active argument without duals contributes nothing
        Cfix = rand(Tensor{2, 3})
        @test gradient(F -> myg(F, Cfix), F0) ≈ gradient(F -> tr(F ⋅ Cfix), F0)
        # duals from different differentiations must not be combined
        x1 = Tensors._load(rand(Tensor{2, 3}), ForwardDiff.Tag(sin, Float64))
        x2 = Tensors._load(rand(Tensor{2, 3}), ForwardDiff.Tag(cos, Float64))
        @test_throws ArgumentError propagate_gradient(myg_dg, Val((1, 2)), x1, x2)
        # one Jacobian per active argument is required
        bad_dg(F, C) = (myg(F, C), (transpose(C),))
        @test_throws ArgumentError propagate_gradient(bad_dg, Val((1, 2)), x1, x1)
    end

    @testset "@tensorop macro" begin
        Tensors.@tensorop function my_otimes(A::Vec{dim}, B::Vec{dim}) where {dim}
            C[i, j] = A[i] * B[j]
        end
        Tensors.@tensorop function my_dcontract(A::SecondOrderTensor, B::SecondOrderTensor)
            @muladd C = A[i, j] * B[i, j]
        end
        a, b = rand(Vec{3}), rand(Vec{3})
        @test my_otimes(a, b) == otimes(a, b)
        S = rand(SymmetricTensor{2, 3})
        @test my_dcontract(S, S) ≈ dcontract(S, S)
        @test (@inferred my_otimes(a, b)) isa Tensor{2, 3, Float64}
        # single-argument declarations (permutations)
        Tensors.@tensorop function my_transpose(A::SecondOrderTensor)
            C[i, j] = A[j, i]
        end
        A2 = rand(Tensor{2, 3})
        @test (@inferred my_transpose(A2))::Tensor{2, 3} == transpose(A2)
        @test my_transpose(S) isa SymmetricTensor{2, 3} # (j,i) carried by S
        m23 = rand(MixedTensor{2, Tuple{2, 3}})
        @test (@inferred my_transpose(m23))::MixedTensor{2, Tuple{3, 2}} == m23'
        # forced output type via left-hand-side annotation
        Tensors.@tensorop function my_truncate(A::Tensor{2, dim}) where {dim}
            C[i, j]::SymmetricTensor{2, dim} = A[i, j]
        end
        @test (@inferred my_truncate(A2))::SymmetricTensor{2, 3} == Tensors.unsafe_symmetric(A2)
    end

    @testset "broadcast lattice" begin
        S = rand(Tensor{2, 2}); Ss = rand(SymmetricTensor{2, 2}); v = rand(Vec{3})
        @test (S .+ 1.0) isa Tensor{2, 2, Float64}
        @test Array(S .+ 1.0) == Array(S) .+ 1.0
        @test (Ss .* 2) isa Tensor{2, 2, Float64}      # general f: densified
        @test (S .+ Ss) isa Tensor{2, 2, Float64}
        @test sqrt.(v) isa Vec{3, Float64}
        @test (S .+ rand(2, 2)) isa Matrix{Float64}    # ordinary arrays win
        @test (v .+ rand(Tensor{2, 3})) isa Matrix{Float64}  # shape mismatch -> array semantics
        @test ((x -> "s").(v)) isa Vector{String}      # non-Number eltype -> Array
        m = rand(MixedTensor{2, Tuple{2, 3}})
        @test (m .* 2) isa MixedTensor{2, Tuple{2, 3}, Float64}
        @test (@inferred (S .* 2.0)) isa Tensor{2, 2, Float64}
    end

    @testset "non-literal integer powers (issue #239)" begin
        for S in (rand(Tensor{2, 3}), rand(SymmetricTensor{2, 3}))
            @test S^3 == S^Int(3)
            @test typeof(S^Int(3)) == typeof(S)
            @test S^-2 == S^Int(-2)
            @test S^Int(0) == one(S)
            @test (@inferred S^Int(2)) isa typeof(S)
            # exponentiation-by-squaring path
            Sn = S / norm(S)
            @test Sn^Int(9) ≈ foldl((a, _) -> Tensors._powdot(a, Sn), 1:8; init = Sn)
            @test typeof(Sn^Int(9)) == typeof(Sn)
        end
        @test_throws ArgumentError rand(MixedTensor{2, Tuple{2, 3}})^Int(2)
    end

    @testset "propagate_gradient / extract_value" begin
        # analytical rule with a passive parameter (issue #197)
        myf(x, p) = p * tr(x) * x
        myf_dfdx(x, p) = (myf(x, p), p * (x ⊗ one(x) + tr(x) * one(SymmetricTensor{4, 3})))
        myf(x::SymmetricTensor{2, 3, <:ForwardDiff.Dual}, p) = propagate_gradient(myf_dfdx, x, p)
        x = rand(SymmetricTensor{2, 3})
        @test gradient(y -> myf(y, 2.5), x) ≈ 2.5 * (x ⊗ one(x) + tr(x) * one(SymmetricTensor{4, 3}))
        # Val{i} form: differentiation argument in a non-leading position
        myg(p, x) = myf(x, p)
        myg_dgdx(p, x) = (myg(p, x), p * (x ⊗ one(x) + tr(x) * one(SymmetricTensor{4, 3})))
        myg(p, x::SymmetricTensor{2, 3, <:ForwardDiff.Dual}) = propagate_gradient(myg_dgdx, Val(2), p, x)
        @test gradient(y -> myg(2.5, y), x) ≈ gradient(y -> myf(y, 2.5), x)
        # external (non-Tensors) ForwardDiff tag: the analytical rule must not
        # depend on the Tag payload matching the input type
        myh(y) = det(y) * y
        myh_dfdx(y) = (myh(y), y ⊗ (det(y) * inv(y)') + det(y) * one(Tensor{4, 3}))
        myh(y::Tensor{2, 3, <:ForwardDiff.Dual}) = propagate_gradient(myh_dfdx, y)
        A = rand(Tensor{2, 3})
        d = ForwardDiff.derivative(t -> norm(myh(t * A)), 1.0)
        d_ref = ForwardDiff.derivative(t -> norm(det(t * A) * (t * A)), 1.0)
        @test d ≈ d_ref
        # nested duals: hessian through an analytical gradient
        @test hessian(y -> norm(myh(y)), A) ≈ hessian(y -> norm(det(y) * y), A)
        # extract_value (issue #208)
        seen = Ref(zero(x))
        g(y) = (seen[] = extract_value(y); norm(y))
        gradient(g, x)
        @test seen[] == x
        @test extract_value(1.23) === 1.23
        @test extract_value(x) === x
    end

    @testset "MixedTensor promote/convert" begin
        m64 = rand(MixedTensor{2, Tuple{2, 3}, Float64})
        m32 = rand(MixedTensor{2, Tuple{2, 3}, Float32})
        p = promote(m64, m32)
        @test p isa NTuple{2, MixedTensor{2, Tuple{2, 3}, Float64, 6}}
        @test p[2] == convert(MixedTensor{2, Tuple{2, 3}, Float64}, m32)
        @test convert(typeof(m64), m64) === m64
    end
end
