SUITE["dot"] = BenchmarkGroup()
SUITE["dcontract"] = BenchmarkGroup()
SUITE["otimes"] = BenchmarkGroup()
SUITE["other"] = BenchmarkGroup()
SUITE["promotion"] = BenchmarkGroup()
SUITE["constructors"] = BenchmarkGroup()
SUITE["basic-operations"] = BenchmarkGroup()


for dim in (ALL_DIMENSIONS ? (1,2,3) : (3,))
    for T in (Float64, Float32, dT)
        v1 = tensor_dict[(dim, 1, T)]
        V2 = tensor_dict[(dim, 2, T)]
        V4 = tensor_dict[(dim, 4, T)]
        V2sym = symtensor_dict[(dim, 2, T)]
        V4sym = symtensor_dict[(dim, 4, T)]

        # dot
        SUITE["dot"]["Vec{$dim, $T} ⋅ Vec{$dim, $T}"]                               = @benchmarkable dot($v1, $v1)
        SUITE["dot"]["Tensor{2, $dim, $T} ⋅ Vec{$dim, $T}"]                         = @benchmarkable dot($V2, $v1)
        SUITE["dot"]["Vec{$dim, $T} ⋅ Tensor{2, $dim, $T}"]                         = @benchmarkable dot($v1, $V2)
        SUITE["dot"]["SymmetricTensor{2, $dim, $T} ⋅ Vec{$dim, $T}"]                = @benchmarkable dot($V2sym, $v1)
        SUITE["dot"]["Vec{$dim, $T} ⋅ SymmetricTensor{2, $dim, $T}"]                = @benchmarkable dot($v1, $V2sym)
        SUITE["dot"]["Tensor{2, $dim, $T} ⋅ Tensor{2, $dim, $T}"]                   = @benchmarkable dot($V2, $V2)
        SUITE["dot"]["SymmetricTensor{2, $dim, $T} ⋅ SymmetricTensor{2, $dim, $T}"] = @benchmarkable dot($V2sym, $V2sym)
        if MIXED_SYM_NONSYM
            SUITE["dot"]["SymmetricTensor{2, $dim, $T} ⋅ Tensor{2, $dim, $T}"] = @benchmarkable dot($V2sym, $V2)
            SUITE["dot"]["Tensor{2, $dim, $T} ⋅ SymmetricTensor{2, $dim, $T}"] = @benchmarkable dot($V2, $V2sym)
        end

        # dcontract
        SUITE["dcontract"]["Tensor{2, $dim, $T} ⊡ Tensor{2, $dim, $T}"]                   = @benchmarkable dcontract($V2, $V2)
        SUITE["dcontract"]["SymmetricTensor{2, $dim, $T} ⊡ SymmetricTensor{2, $dim, $T}"] = @benchmarkable dcontract($V2sym, $V2sym)
        if MIXED_SYM_NONSYM
            SUITE["dcontract"]["Tensor{2, $dim, $T} ⊡ SymmetricTensor{2, $dim, $T}"] = @benchmarkable dcontract($V2, $V2sym)
            SUITE["dcontract"]["SymmetricTensor{2, $dim, $T} ⊡ Tensor{2, $dim, $T}"] = @benchmarkable dcontract($V2sym, $V2)
        end
        SUITE["dcontract"]["Tensor{4, $dim, $T} ⊡ Tensor{2, $dim, $T}"] = @benchmarkable dcontract($V4, $V2)
        SUITE["dcontract"]["Tensor{2, $dim, $T} ⊡ Tensor{4, $dim, $T}"] = @benchmarkable dcontract($V2, $V4)
        SUITE["dcontract"]["Tensor{4, $dim, $T} ⊡ Tensor{4, $dim, $T}"] = @benchmarkable dcontract($V4, $V4)
        if MIXED_SYM_NONSYM
            SUITE["dcontract"]["SymmetricTensor{4, $dim, $T} ⊡ Tensor{2, $dim, $T}"] = @benchmarkable dcontract($V4sym, $V2)
            SUITE["dcontract"]["Tensor{2, $dim, $T} ⊡ SymmetricTensor{4, $dim, $T}"] = @benchmarkable dcontract($V2, $V4sym)
            SUITE["dcontract"]["Tensor{4, $dim, $T} ⊡ SymmetricTensor{2, $dim, $T}"] = @benchmarkable dcontract($V4, $V2sym)
            SUITE["dcontract"]["SymmetricTensor{2, $dim, $T} ⊡ Tensor{4, $dim, $T}"] = @benchmarkable dcontract($V2sym, $V4)
            SUITE["dcontract"]["SymmetricTensor{4, $dim, $T} ⊡ Tensor{4, $dim, $T}"] = @benchmarkable dcontract($V4sym, $V4)
            SUITE["dcontract"]["Tensor{4, $dim, $T} ⊡ SymmetricTensor{4, $dim, $T}"] = @benchmarkable dcontract($V4, $V4sym)
        end
        SUITE["dcontract"]["Tensor{4, $dim, $T} ⊡ Tensor{4, $dim, $T}"]                   = @benchmarkable dcontract($V4, $V4)
        SUITE["dcontract"]["SymmetricTensor{4, $dim, $T} ⊡ SymmetricTensor{4, $dim, $T}"] = @benchmarkable dcontract($V4sym, $V4sym)

        # otimes
        SUITE["otimes"]["Vec{$dim, $T} ⊗ Vec{$dim, $T}"]                               = @benchmarkable otimes($v1, $v1)
        SUITE["otimes"]["Tensor{2, $dim, $T} ⊗ Tensor{2, $dim, $T}"]                   = @benchmarkable otimes($V2, $V2)
        SUITE["otimes"]["SymmetricTensor{2, $dim, $T} ⊗ SymmetricTensor{2, $dim, $T}"] = @benchmarkable otimes($V2sym, $V2sym)
        if MIXED_SYM_NONSYM
            SUITE["otimes"]["Tensor{2, $dim, $T} ⊗ SymmetricTensor{2, $dim, $T}"] = @benchmarkable otimes($V2, $V2sym)
            SUITE["otimes"]["SymmetricTensor{2, $dim, $T} ⊗ Tensor{2, $dim, $T}"] = @benchmarkable otimes($V2sym, $V2)
        end

        # other
        for (i, V2t) in enumerate((V2, V2sym))
            TensorType = i == 2 ? "SymmetricTensor" : "Tensor"
            for f in (norm, tr, vol, det, inv, transpose, symmetric, skew, eig, mean, dev)
                (i == 1 || typeof(V2t) <: Tensor || T == dT) && f == eig && continue
                SUITE["other"]["$f($TensorType{2, $dim, $T})"] = @benchmarkable $f($V2t)
            end
        end

        for (i, V4t) in enumerate((V4, V4sym))
            TensorType = i == 2 ? "SymmetricTensor" : "Tensor"
            for f in (norm, symmetric)
                SUITE["other"]["$f($TensorType{4, $dim, $T})"] = @benchmarkable $f($V4t)
            end
        end

        if T in (Float32, Float64)
            for f in (:+, :-)
                SUITE["basic-operations"]["Vec{$dim, $T} $f Vec{$dim, $T}"]                               = @benchmarkable $f($v1, $v1)
                SUITE["basic-operations"]["Tensor{2, $dim, $T} $f Tensor{2, $dim, $T}"]                   = @benchmarkable $f($V2, $V2)
                SUITE["basic-operations"]["SymmetricTensor{2, $dim, $T} $f SymmetricTensor{2, $dim, $T}"] = @benchmarkable $f($V2sym, $V2sym)
                SUITE["basic-operations"]["Tensor{4, $dim, $T} $f Tensor{4, $dim, $T}"]                   = @benchmarkable $f($V4, $V4)
                SUITE["basic-operations"]["SymmetricTensor{4, $dim, $T} $f SymmetricTensor{4, $dim, $T}"] = @benchmarkable $f($V4sym, $V4sym)
                if MIXED_SYM_NONSYM
                    SUITE["basic-operations"]["Tensor{2, $dim, $T} $f SymmetricTensor{2, $dim, $T}"]      = @benchmarkable $f($V2, $V2sym)
                    SUITE["basic-operations"]["Tensor{4, $dim, $T} $f SymmetricTensor{4, $dim, $T}"]      = @benchmarkable $f($V4, $V4sym)
                end
            end
            for f in (:*, :/)
                n = rand(T)
                SUITE["basic-operations"]["Vec{$dim, $T} $f $T"]                = @benchmarkable $f($v1, $n)
                SUITE["basic-operations"]["Tensor{2, $dim, $T} $f $T"]          = @benchmarkable $f($V2, $n)
                SUITE["basic-operations"]["SymmetricTensor{2, $dim, $T} $f $T"] = @benchmarkable $f($V2sym, $n)
                SUITE["basic-operations"]["Tensor{4, $dim, $T} $f $T"]          = @benchmarkable $f($V4, $n)
                SUITE["basic-operations"]["SymmetricTensor{4, $dim, $T} $f $T"] = @benchmarkable $f($V4sym, $n)
                if f == :*
                    SUITE["basic-operations"]["$T $f Vec{$dim, $T}"]                = @benchmarkable $f($n, $v1)
                    SUITE["basic-operations"]["$T $f Tensor{2, $dim, $T}"]          = @benchmarkable $f($n, $V2)
                    SUITE["basic-operations"]["$T $f SymmetricTensor{2, $dim, $T}"] = @benchmarkable $f($n, $V2sym)
                    SUITE["basic-operations"]["$T $f Tensor{4, $dim, $T}"]          = @benchmarkable $f($n, $V4)
                    SUITE["basic-operations"]["$T $f SymmetricTensor{4, $dim, $T}"] = @benchmarkable $f($n, $V4sym)
                end
            end
        end
    end
end

for dim in (ALL_DIMENSIONS ? (1,2,3) : (3,))
    a32 = tensor_dict[(dim, 1, Float32)]
    a64 = tensor_dict[(dim, 1, Float64)]
    A32 = tensor_dict[(dim, 2, Float32)]
    A64 = tensor_dict[(dim, 2, Float64)]
    AA32 = tensor_dict[(dim, 4, Float32)]
    AA64 = tensor_dict[(dim, 4, Float64)]
    A32s = symtensor_dict[(dim, 2, Float32)]
    A64s = symtensor_dict[(dim, 2, Float64)]
    AA32s = symtensor_dict[(dim, 4, Float32)]
    AA64s = symtensor_dict[(dim, 4, Float64)]

    # promotion between tensortypes
    SUITE["promotion"]["SymmetricTensor{2, $dim} -> Tensor{2, $dim}"] = @benchmarkable promote($A64, $A64s)
    SUITE["promotion"]["SymmetricTensor{4, $dim} -> Tensor{4, $dim}"] = @benchmarkable promote($AA64, $AA64s)

    # element type promotion
    SUITE["promotion"]["Vec{dim, Float32} -> Vec{$dim, Float64}"]                                = @benchmarkable promote($a64, $a32)
    SUITE["promotion"]["Tensor{2, $dim, Float32} -> Tensor{2, $dim, Float64}"]                   = @benchmarkable promote($A64, $A32)
    SUITE["promotion"]["SymmetricTensor{2, $dim, Float32} -> SymmetricTensor{2, $dim, Float64}"] = @benchmarkable promote($A64s, $A32s)
    SUITE["promotion"]["Tensor{4, $dim, Float32} -> Tensor{4, $dim, Float64}"]                   = @benchmarkable promote($A64, $A32)
    SUITE["promotion"]["SymmetricTensor{4, $dim, Float32} -> SymmetricTensor{4, $dim, Float64}"] = @benchmarkable promote($A64s, $A32s)

    # test just some operations with mixed tensortype and eltype and hope it catches things like
    # https://github.com/Ferrite-FEM/Tensors.jl/pull/5#issuecomment-282518974
    if MIXED_ELTYPES
        n = 5
        SUITE["promotion"]["Tensor{2, $dim, Float64} * $Int"]                     = @benchmarkable *($A64, $n)
        SUITE["promotion"]["Tensor{2, $dim, Float64} / $Int"]                     = @benchmarkable /($A64, $n)
        SUITE["promotion"]["Tensor{2, $dim, Float32} + Tensor{2, $dim, Float64}"] = @benchmarkable +($A32, $A64)
        SUITE["promotion"]["Tensor{2, $dim, Float32} - Tensor{2, $dim, Float64}"] = @benchmarkable -($A32, $A64)
        SUITE["promotion"]["Tensor{2, $dim, Float32} ⋅ Tensor{2, $dim, Float64}"] = @benchmarkable dot($A32, $A64)
        SUITE["promotion"]["Tensor{2, $dim, Float32} ⊡ Tensor{2, $dim, Float64}"] = @benchmarkable dcontract($A32, $A64)
        SUITE["promotion"]["Tensor{2, $dim, Float32} ⊗ Tensor{2, $dim, Float64}"] = @benchmarkable otimes($A32, $A64)
    end
end

# constructors (only testing for dim = 3)
# could be done cleaner, but https://github.com/JuliaCI/BenchmarkTools.jl/issues/50
for f in (:zero, :one, :ones, :rand)
    SUITE["constructors"]["$f(Tensor{2, 3, Float32})"]          = @benchmarkable $(f)(Tensor{2, 3, Float32})
    SUITE["constructors"]["$f(Tensor{4, 3, Float32})"]          = @benchmarkable $(f)(Tensor{4, 3, Float32})
    SUITE["constructors"]["$f(SymmetricTensor{2, 3, Float32})"] = @benchmarkable $(f)(SymmetricTensor{2, 3, Float32})
    SUITE["constructors"]["$f(SymmetricTensor{4, 3, Float32})"] = @benchmarkable $(f)(SymmetricTensor{4, 3, Float32})
    SUITE["constructors"]["$f(Tensor{2, 3, Float64})"]          = @benchmarkable $(f)(Tensor{2, 3, Float64})
    SUITE["constructors"]["$f(Tensor{4, 3, Float64})"]          = @benchmarkable $(f)(Tensor{4, 3, Float64})
    SUITE["constructors"]["$f(SymmetricTensor{2, 3, Float64})"] = @benchmarkable $(f)(SymmetricTensor{2, 3, Float64})
    SUITE["constructors"]["$f(SymmetricTensor{4, 3, Float64})"] = @benchmarkable $(f)(SymmetricTensor{4, 3, Float64})
    if f != :one
        SUITE["constructors"]["$f(Vec{3, Float32})"] = @benchmarkable $(f)(Vec{3, Float32})
        SUITE["constructors"]["$f(Vec{3, Float64})"] = @benchmarkable $(f)(Vec{3, Float64})
    end
end

# create from a Julia array
for order in (1, 2, 4)
    dim = 3
    A = rand(Tensors.n_components(Tensor{order, dim}))
    SUITE["constructors"]["Tensor{$order, $dim}(A::Array)"] = @benchmarkable Tensor{$order, $dim}($A)
    if order != 1
        As = rand(Tensors.n_components(SymmetricTensor{order, dim}))
        SUITE["constructors"]["SymmetricTensor{$order, $dim}(A::Array)"] = @benchmarkable SymmetricTensor{$order, $dim}($As)
    end
end

begin
    dim = 3
    f1 = (i) -> float(i)
    f2 = (i, j) -> float(i)
    f4 = (i, j, k, l) -> float(i)
    SUITE["constructors"]["Vec{3}(f::Function)"] =       @benchmarkable Vec{3}($f1)
    SUITE["constructors"]["Tensor{2, 3}(f::Function)"] = @benchmarkable Tensor{2, 3}($f2)
    SUITE["constructors"]["Tensor{4, 3}(f::Function)"] = @benchmarkable Tensor{4, 3}($f4)
    SUITE["constructors"]["SymmetricTensor{2, 3}(f::Function)"] = @benchmarkable SymmetricTensor{2, 3}($f2)
    SUITE["constructors"]["SymmetricTensor{4, 3}(f::Function)"] = @benchmarkable SymmetricTensor{4, 3}($f4)
end
