SUITE["dot"] = BenchmarkGroup()
SUITE["dcontract"] = BenchmarkGroup()
SUITE["otimes"] = BenchmarkGroup()
SUITE["other"] = BenchmarkGroup()
SUITE["promotion"] = BenchmarkGroup()


for dim in (1,2,3)
    for T in (Float64, Float32)
        v1 = tensor_dict[(dim, 1, T)]
        V2 = tensor_dict[(dim, 2, T)]
        V4 = tensor_dict[(dim, 4, T)]
        V2sym = symtensor_dict[(dim, 2, T)]
        V4sym = symtensor_dict[(dim, 4, T)]

        # dot
        SUITE["dot"]["dim $dim - order 1 1 - $T"]             = @benchmarkable dot($v1, $v1)
        SUITE["dot"]["dim $dim - order 2 1 - $T"]             = @benchmarkable dot($V2, $v1)
        SUITE["dot"]["dim $dim - order 2sym 1 - $T"]          = @benchmarkable dot($V2sym, $v1)
        SUITE["dot"]["dim $dim - order 2 2 - $T"]             = @benchmarkable dot($V2, $V2)
        SUITE["dot"]["dim $dim - order 2sym 2sym - $T"]       = @benchmarkable dot($V2sym, $V2sym)
        SUITE["dot"]["dim $dim - order 2sym 2 - $T"]          = @benchmarkable dot($V2sym, $V2)

        # dcontract
        SUITE["dcontract"]["dim $dim - order 2 2 - $T"]       = @benchmarkable dcontract($V2, $V2)
        SUITE["dcontract"]["dim $dim - order 2sym 2sym - $T"] = @benchmarkable dcontract($V2sym, $V2sym)
        SUITE["dcontract"]["dim $dim - order 2sym 2 - $T"]    = @benchmarkable dcontract($V2sym, $V2)

        SUITE["dcontract"]["dim $dim - order 4 2 - $T"]       = @benchmarkable dcontract($V4, $V2)
        SUITE["dcontract"]["dim $dim - order 4sym 2 - $T"]    = @benchmarkable dcontract($V4sym, $V2)
        SUITE["dcontract"]["dim $dim - order 2 4sym - $T"]    = @benchmarkable dcontract($V2, $V4sym)
        SUITE["dcontract"]["dim $dim - order 2 4 - $T"]       = @benchmarkable dcontract($V2, $V4)
        SUITE["dcontract"]["dim $dim - order 4 2sym - $T"]    = @benchmarkable dcontract($V4, $V2sym)
        SUITE["dcontract"]["dim $dim - order 4sym 2sym - $T"] = @benchmarkable dcontract($V4, $V2sym)

        SUITE["dcontract"]["dim $dim - order 4 4 - $T"]       = @benchmarkable dcontract($V4, $V4)
        SUITE["dcontract"]["dim $dim - order 4sym 4 - $T"]    = @benchmarkable dcontract($V4sym, $V4)
        SUITE["dcontract"]["dim $dim - order 4sym 4sym - $T"] = @benchmarkable dcontract($V4sym, $V4sym)

        # otimes
        SUITE["otimes"]["dim $dim - order 1 1 - $T"]          = @benchmarkable otimes($v1, $v1)
        SUITE["otimes"]["dim $dim - order 2 2 - $T"]          = @benchmarkable otimes($V2, $V2)
        SUITE["otimes"]["dim $dim - order 2sym 2sym - $T"]    = @benchmarkable otimes($V2sym, $V2sym)
        SUITE["otimes"]["dim $dim - order 2sym 2 - $T"]       = @benchmarkable otimes($V2sym, $V2)

        # other
        for (i, V2t) in enumerate((V2, V2sym))
            symstr = i == 2 ? "sym" : ""
            for f in (norm, trace, vol, det, inv, transpose, symmetric, skew, eig, mean, dev)
                i == 1 && f == eig && continue
                SUITE["other"]["$f - dim $dim - order 2$(symstr) - $T"] = @benchmarkable $f($V2t)
            end
        end

        for (i, V4t) in enumerate((V4, V4sym))
            symstr = i == 2 ? "sym" : ""
            for f in (norm, symmetric)
                SUITE["other"]["$f - dim $dim - order 4$(symstr) - $T"] = @benchmarkable $f($V4t)
            end
        end

        SUITE["promotion"]["dim $dim - order 2"] = @benchmarkable promote($V2, $V2sym)
        SUITE["promotion"]["dim $dim - order 4"] = @benchmarkable promote($V4, $V4sym)
    end
end
