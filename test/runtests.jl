# Build the real docs first.
include("../docs/make.jl")

using ContMechTensors
using Base.Test

include("test_misc.jl")
include("test_ops.jl")
include("test_ad.jl")
