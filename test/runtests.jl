using Tensors
using ParallelTestRunner

testsuite = find_tests(@__DIR__)
delete!(testsuite, "F64") # helper definitions, loaded in every test via init_code

init_code = quote
    using Tensors
    using Test
    using LinearAlgebra
    using Random
    using Statistics: mean
    using StaticArrays
    import ForwardDiff
    include($(joinpath(@__DIR__, "F64.jl")))
end

runtests(Tensors, ARGS; testsuite, init_code)
