# ContMechTensors

[![Build Status](https://travis-ci.org/KristofferC/ContMechTensors.jl.svg?branch=master)](https://travis-ci.org/KristofferC/ContMechTensors.jl) [![codecov.io](https://codecov.io/github/KristofferC/ContMechTensors.jl/coverage.svg?branch=master)](https://codecov.io/github/KristofferC/ContMechTensors.jl?branch=master)




### Creating Tensors




### Indexing


### Operations





### Indexing

Indexing into a `Tensor` is simply done with an index from 1 to `dim`

```jl
# Create a random second order tensor
julia> A = rand(Tensor{2});

julia> A[1,2]

julia> A[:x, :y]
```

### `one`, `rand`, `zero`



```jl
julia> A = rand(Tensor{1, 2})


julia> B = rand(SymmetricTensor{2, 3}, Float32)


julia> C = one(B)



# Operations



