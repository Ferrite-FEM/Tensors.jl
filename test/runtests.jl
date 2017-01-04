using ContMechTensors
using Base.Test

include("test_misc.jl")
include("test_ops.jl")
include("test_ad.jl")

# Build the docs
include("../docs/make.jl")
