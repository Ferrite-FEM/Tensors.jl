# MWE for wrong results from `dott` (Tensors.jl) on Julia 1.13.0-rc1 on some
# x86-64 machines (observed on GitHub Actions ubuntu-latest, machine-dependent).
#
# Two levels:
#   1. "SIMD-level": self-contained reduction using only SIMD.jl, mirroring
#      Tensors.jl's `dott(A) = unsafe_symmetric(A ⋅ A')` after inlining:
#      materialize transpose -> SIMD muladd dot -> extract lower triangle.
#   2. "Tensors-level": the original failing operation via Tensors.jl itself.
#
# Exits with code 1 if any wrong result is found.

import Pkg
Pkg.activate(; temp=true, io=devnull)
Pkg.add("SIMD"; io=devnull)
tensors_path = get(ENV, "TENSORS_PATH", "")
have_tensors = !isempty(tensors_path)
have_tensors && Pkg.develop(path=tensors_path, io=devnull)

using InteractiveUtils
versioninfo(; verbose=true)
println("CPU_NAME: ", Sys.CPU_NAME)
println("check-bounds: ", Base.JLOptions().check_bounds)
println()

using SIMD

# Mirrors Tensors.jl: transpose materialization + SIMD dot (src/simd.jl:174)
# + unsafe_symmetric lower-triangle extraction, all inlined into one function.
@inline function dott_simd(D1::NTuple{9, T}) where {T}
    @inbounds begin
        # transpose(A) materialized as a new tuple (Tensors.transpose)
        D2 = (D1[1], D1[4], D1[7], D1[2], D1[5], D1[8], D1[3], D1[6], D1[9])
        # dot(A, At): three column SVecs of A, scalar broadcasts from At
        SV11 = Vec{3, T}((D1[1], D1[2], D1[3]))
        SV12 = Vec{3, T}((D1[4], D1[5], D1[6]))
        SV13 = Vec{3, T}((D1[7], D1[8], D1[9]))
        r1 = muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))
        r2 = muladd(SV13, D2[6], muladd(SV12, D2[5], SV11 * D2[4]))
        r3 = muladd(SV13, D2[9], muladd(SV12, D2[8], SV11 * D2[7]))
        # Tensor{2,3} from NTuple{3, SVec{3}} (src/simd.jl:62)
        full = (r1[1], r1[2], r1[3], r2[1], r2[2], r2[3], r3[1], r3[2], r3[3])
        # unsafe_symmetric: keep lower triangle (1,1),(2,1),(3,1),(2,2),(3,2),(3,3)
        return (full[1], full[2], full[3], full[5], full[6], full[9])
    end
end

function ref_dott(D1::NTuple{9, T}) where {T}
    A = reshape(collect(D1), 3, 3)
    C = A * A'
    return (C[1, 1], C[2, 1], C[3, 1], C[2, 2], C[3, 2], C[3, 3])
end

simd_fails = 0
for T in (Float32, Float64), trial in 1:1000
    D = ntuple(_ -> rand(T), 9)
    got = dott_simd(D)
    want = ref_dott(D)
    if !all(isapprox.(got, want; rtol = sqrt(eps(T))))
        global simd_fails += 1
        if simd_fails <= 5
            println("SIMD-level FAIL (T = $T):")
            println("  got  = ", got)
            println("  want = ", want)
        end
    end
end
println(simd_fails == 0 ? "SIMD-level MWE: PASS" : "SIMD-level MWE: FAIL ($simd_fails cases)")

tensors_fails = 0
if have_tensors
    using Tensors
    for T in (Float32, Float64), trial in 1:1000
        A = rand(Tensor{2, 3, T})
        d = dott(A)
        ref = SymmetricTensor{2, 3}((i, j) -> sum(A[i, k] * A[j, k] for k in 1:3))
        if !isapprox(d, ref; rtol = sqrt(eps(T)))
            global tensors_fails += 1
            if tensors_fails <= 5
                println("Tensors-level FAIL (T = $T):")
                display(d)
                display(ref)
            end
        end
    end
    println(tensors_fails == 0 ? "Tensors-level dott: PASS" : "Tensors-level dott: FAIL ($tensors_fails cases)")
end

(simd_fails > 0 || tensors_fails > 0) && exit(1)
