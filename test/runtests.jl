using Tensors
using Test
using TimerOutputs
using LinearAlgebra
using Random
using Statistics: mean
using StaticArrays

macro testsection(str, block)
    return quote
        @timeit "$($(esc(str)))" begin
            @testset "$($(esc(str)))" begin
                $(esc(block))
            end
        end
    end
end

reset_timer!()

include("test_dotmacro.jl")
include("F64.jl")
include("test_misc.jl")
include("test_ops.jl")
include("test_ad.jl")

print_timer()
println()
