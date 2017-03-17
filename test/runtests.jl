using Tensors
using Base.Test
using TimerOutputs

macro testsection(args...)
    name = esc(args[1])
    return quote
        @timeit $name begin
            @testset($(map(esc, args)...))
        end
    end
end

reset_timer!()

include("F64.jl")
include("test_misc.jl")
include("test_ops.jl")
include("test_ad.jl")

print_timer()
println()

# Build the docs
include("../docs/make.jl")
