# MWE for wrong results from `dott` (Tensors.jl) on Julia 1.13.0-rc1 on x86-64
# machines with AVX-512 (GitHub Actions ubuntu-latest, machine-dependent).
#
# The 5th element of the returned 6-tuple duplicates the 4th element.
# Key observation: the bug only triggers when the kernel is invoked via
# dynamic dispatch from top level, not when inlined into a compiled caller.
#
# Variants:
#   A) pure-Base kernel, dynamic top-level call
#   B) pure-Base kernel, loop inside a function (kernel inlined)
#   C) pure-Base kernel behind @noinline, called from a compiled function
#      (standalone sret ABI, but no jfptr/boxing wrapper)
#   D) SIMD.jl kernel, dynamic top-level call (original reduction)
#   E) Tensors.jl dott, dynamic top-level call (original failure)
#
# Exits with code 1 if any wrong result is found.

using InteractiveUtils
versioninfo(; verbose = true)
println("CPU_NAME: ", Sys.CPU_NAME)
println("check-bounds: ", Base.JLOptions().check_bounds)
println()

# ------------------------------------------------------------------
# Pure-Base kernel (no packages)
# ------------------------------------------------------------------
module K

const VE = Core.VecElement
const V3 = NTuple{3, VE{Float32}}

@inline vmul(a::V3, b::V3) = Core.Intrinsics.llvmcall(
    "%r = fmul <3 x float> %0, %1\nret <3 x float> %r",
    V3, Tuple{V3, V3}, a, b)
@inline vfmuladd(a::V3, b::V3, c::V3) = ccall("llvm.fmuladd.v3f32", llvmcall,
    V3, (V3, V3, V3), a, b, c)
@inline vbc(x::Float32) = (VE(x), VE(x), VE(x))

@inline function dott_kernel(D1::NTuple{9, Float32})
    D2 = (D1[1], D1[4], D1[7], D1[2], D1[5], D1[8], D1[3], D1[6], D1[9])
    SV11 = (VE(D1[1]), VE(D1[2]), VE(D1[3]))
    SV12 = (VE(D1[4]), VE(D1[5]), VE(D1[6]))
    SV13 = (VE(D1[7]), VE(D1[8]), VE(D1[9]))
    r1 = vfmuladd(SV13, vbc(D2[3]), vfmuladd(SV12, vbc(D2[2]), vmul(SV11, vbc(D2[1]))))
    r2 = vfmuladd(SV13, vbc(D2[6]), vfmuladd(SV12, vbc(D2[5]), vmul(SV11, vbc(D2[4]))))
    r3 = vfmuladd(SV13, vbc(D2[9]), vfmuladd(SV12, vbc(D2[8]), vmul(SV11, vbc(D2[7]))))
    return (r1[1].value, r1[2].value, r1[3].value,
            r2[2].value, r2[3].value,
            r3[3].value)
end

@noinline dott_kernel_noinline(D1::NTuple{9, Float32}) = dott_kernel(D1)

function ref(D1::NTuple{9, Float32})
    A = reshape(collect(D1), 3, 3)
    C = A * A'
    return (C[1, 1], C[2, 1], C[3, 1], C[2, 2], C[3, 2], C[3, 3])
end

check(got, want) = all(isapprox.(got, want; rtol = sqrt(eps(Float32))))

function run_inlined(n)
    fails = 0
    for _ in 1:n
        D = ntuple(_ -> rand(Float32), 9)
        check(dott_kernel(D), ref(D)) || (fails += 1)
    end
    return fails
end

function run_noinline(n)
    fails = 0
    for _ in 1:n
        D = ntuple(_ -> rand(Float32), 9)
        check(dott_kernel_noinline(D), ref(D)) || (fails += 1)
    end
    return fails
end

end # module K

report(label, fails, n) = println(rpad(label, 55), fails == 0 ? "PASS" : "FAIL ($fails/$n)")
N = 500

# A) dynamic top-level calls
fails_A = 0
example_A = nothing
for _ in 1:N
    D = ntuple(_ -> rand(Float32), 9)
    got = K.dott_kernel(D)
    want = K.ref(D)
    if !K.check(got, want)
        global fails_A += 1
        global example_A = (D, got, want)
    end
end
report("A) pure-Base, dynamic top-level call", fails_A, N)
if example_A !== nothing
    println("   example D    = ", example_A[1])
    println("   example got  = ", example_A[2])
    println("   example want = ", example_A[3])
end

# B) inlined in compiled caller
fails_B = K.run_inlined(N)
report("B) pure-Base, inlined in compiled function", fails_B, N)

# C) @noinline kernel called from compiled caller (sret ABI, no jfptr)
fails_C = K.run_noinline(N)
report("C) pure-Base, @noinline callee from compiled function", fails_C, N)

# C2) @noinline kernel called dynamically from top level
fails_C2 = 0
for _ in 1:N
    D = ntuple(_ -> rand(Float32), 9)
    K.check(K.dott_kernel_noinline(D), K.ref(D)) || (global fails_C2 += 1)
end
report("C2) pure-Base @noinline, dynamic top-level call", fails_C2, N)

# ------------------------------------------------------------------
# Dumps of the kernel in standalone form (module dump includes the
# jfptr wrapper used by dynamic dispatch). No execution involved.
# ------------------------------------------------------------------
if get(ENV, "MWE_DUMP", "onfail") == "always" || fails_A + fails_B + fails_C + fails_C2 > 0
    println("\n--- code_llvm (dump_module) of dott_kernel_noinline ---")
    code_llvm(stdout, K.dott_kernel_noinline, (NTuple{9, Float32},);
              debuginfo = :none, dump_module = true)
    println("\n--- code_native (dump_module) of dott_kernel_noinline ---")
    code_native(stdout, K.dott_kernel_noinline, (NTuple{9, Float32},);
                debuginfo = :none, dump_module = true)
end

# ------------------------------------------------------------------
# D/E) Original package-level reproductions (need Pkg)
# ------------------------------------------------------------------
fails_D = 0
fails_E = 0
if get(ENV, "MWE_PACKAGES", "true") == "true"
    import Pkg
    Pkg.activate(; temp = true, io = devnull)
    Pkg.add("SIMD"; io = devnull)
    tensors_path = get(ENV, "TENSORS_PATH", "")
    isempty(tensors_path) || Pkg.develop(path = tensors_path, io = devnull)

    using SIMD
    @inline function dott_simd(D1::NTuple{9, T}) where {T}
        @inbounds begin
            D2 = (D1[1], D1[4], D1[7], D1[2], D1[5], D1[8], D1[3], D1[6], D1[9])
            SV11 = Vec{3, T}((D1[1], D1[2], D1[3]))
            SV12 = Vec{3, T}((D1[4], D1[5], D1[6]))
            SV13 = Vec{3, T}((D1[7], D1[8], D1[9]))
            r1 = muladd(SV13, D2[3], muladd(SV12, D2[2], SV11 * D2[1]))
            r2 = muladd(SV13, D2[6], muladd(SV12, D2[5], SV11 * D2[4]))
            r3 = muladd(SV13, D2[9], muladd(SV12, D2[8], SV11 * D2[7]))
            full = (r1[1], r1[2], r1[3], r2[1], r2[2], r2[3], r3[1], r3[2], r3[3])
            return (full[1], full[2], full[3], full[5], full[6], full[9])
        end
    end
    for _ in 1:N
        D = ntuple(_ -> rand(Float32), 9)
        K.check(dott_simd(D), K.ref(D)) || (global fails_D += 1)
    end
    report("D) SIMD.jl kernel, dynamic top-level call", fails_D, N)

    if !isempty(tensors_path)
        using Tensors
        for _ in 1:N
            A = rand(Tensor{2, 3, Float32})
            d = dott(A)
            r = SymmetricTensor{2, 3}((i, j) -> sum(A[i, k] * A[j, k] for k in 1:3))
            isapprox(d, r; rtol = sqrt(eps(Float32))) || (global fails_E += 1)
        end
        report("E) Tensors.jl dott, dynamic top-level call", fails_E, N)
    end
end

total = fails_A + fails_B + fails_C + fails_C2 + fails_D + fails_E
println("\nTOTAL: ", total == 0 ? "all PASS" : "FAILURES present")
total > 0 && exit(1)
