using Tensors
using Compat.Test
using TimerOutputs

macro testsection(str, block)
    return quote
        @timeit $str begin
            @testset $str begin
                println($str)
                $(esc(block))
            end
        end
    end
end

reset_timer!()

#include("F64.jl")
const F64 = Float64
include("test_misc.jl")
include("test_ops.jl")
#include("test_ad.jl")

print_timer()
println()

# Build the docs
#include("../docs/make.jl")
