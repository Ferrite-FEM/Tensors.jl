using Tensors
using BenchmarkTools
using JLD

include("generate_report.jl")

const SUITE = BenchmarkGroup()
const ALL_DIMENSIONS = true
const MIXED_SYM_NONSYM = true
const MIXED_ELTYPES = true

const dT = Tensors.Dual{4, Float64}

function create_tensors()
    tensor_dict = Dict{Tuple{Int, Int, DataType}, AbstractTensor}()
    symtensor_dict = Dict{Tuple{Int, Int, DataType}, AbstractTensor}()
    for dim in 1:3
        for order in (1,2,4)
            for T in (Float32, Float64)
                tensor_dict[(dim, order, T)] = rand(Tensor{order, dim, T})
                if order != 1
                    symtensor_dict[(dim, order, T)] = rand(SymmetricTensor{order, dim, T})
                else
                    symtensor_dict[(dim, order, T)] = rand(Tensor{order, dim, T})
                end
            end
            tensor_dict[(dim, order, dT)] = Tensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...) for i in 1:length(rand(Tensor{order, dim}))]...))
            if order != 1
                symtensor_dict[(dim, order, dT)] = SymmetricTensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...) for i in 1:length(rand(SymmetricTensor{order, dim}).data)]...))
            else
                symtensor_dict[(dim, order, dT)] = Tensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...) for i in 1:length(rand(Tensor{order, dim}).data)]...))
            end
        end
    end
    return tensor_dict, symtensor_dict
end

tensor_dict, symtensor_dict = create_tensors()

include("benchmark_functions.jl")
include("benchmark_ad.jl")

function run_benchmarks(name, tagfilter = @tagged ALL)
    const paramspath = joinpath(dirname(@__FILE__), "params.jld")
    if !isfile(paramspath)
        println("Tuning benchmarks...")
        tune!(SUITE, verbose=true)
        JLD.save(paramspath, "SUITE", params(SUITE))
    end
    loadparams!(SUITE, JLD.load(paramspath, "SUITE"), :evals, :samples)
    results = run(SUITE[tagfilter], verbose = true, seconds = 2)
    JLD.save(joinpath(dirname(@__FILE__), name * ".jld"), "results", results)
end

function generate_report(v1, v2)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    v2_res = load(joinpath(dirname(@__FILE__), v2 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_$(v1)_$(v2).md"), "w") do f
        printreport(f, judge(minimum(v1_res), minimum(v2_res)); iscomparisonjob = true)
    end
end

function generate_report(v1)
    v1_res = load(joinpath(dirname(@__FILE__), v1 * ".jld"), "results")
    open(joinpath(dirname(@__FILE__), "results_$(v1).md"), "w") do f
        printreport(f, minimum(v1_res); iscomparisonjob = false)
    end
end

