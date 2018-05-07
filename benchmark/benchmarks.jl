using Tensors
using BenchmarkTools
using ForwardDiff

const SUITE = BenchmarkGroup()
const ALL_DIMENSIONS = true
const MIXED_SYM_NONSYM = true
const MIXED_ELTYPES = true

const dT = ForwardDiff.Dual{Nothing,Float64,4}

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
            tensor_dict[(dim, order, dT)] = Tensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...,) for i in 1:length(rand(Tensor{order, dim}))]...,))
            if order != 1
                symtensor_dict[(dim, order, dT)] = SymmetricTensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...,) for i in 1:length(rand(SymmetricTensor{order, dim}).data)]...,))
            else
                symtensor_dict[(dim, order, dT)] = Tensor{order, dim, dT}(([ForwardDiff.Dual(rand(5)...,) for i in 1:length(rand(Tensor{order, dim}).data)]...,))
            end
        end
    end
    return tensor_dict, symtensor_dict
end

tensor_dict, symtensor_dict = create_tensors()

include("benchmark_functions.jl")
include("benchmark_ad.jl")
