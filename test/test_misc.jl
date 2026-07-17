@testset "constructors" begin
for T in (Float32, Float64, F64), dim in (1,2,3), order in (1,2,3,4)
    for op in (rand, zero, ones, randn)
        # Tensor, SymmetricTensor
        for TensorType in (Tensor, SymmetricTensor)
            TensorType == SymmetricTensor && isodd(order) && continue
            N = Tensors.n_components(TensorType{order, dim})
            t = (@inferred (op)(TensorType{order, dim}))::TensorType{order, dim, Float64}
            t = (@inferred (op)(TensorType{order, dim, T}))::TensorType{order, dim, T}
            t = (@inferred (op)(TensorType{order, dim, T, N}))::TensorType{order, dim, T}
            t = (@inferred (op)(t))::TensorType{order, dim, T}

            op == zero && @test zero(TensorType{order, dim, T}) == zeros(T, size(t))
            op == ones && @test ones(TensorType{order, dim, T}) == ones(T, size(t))
        end
        # Vec
        if order == 1
            (@inferred (op)(Vec{dim}))::Tensor{order, dim, Float64}
            (@inferred (op)(Vec{dim, T}))::Tensor{order, dim, T}
        end
    end
    # Special Vec constructor
    if order == 1
        t = ntuple(i -> T(i), dim)
        @test (@inferred Vec(t))::Tensor{1,dim,T,dim} == Vec{dim}(t)
        @test (@inferred Vec(t...))::Tensor{1,dim,T,dim} == Vec{dim}(t)
    end
    for TensorType in (Tensor, SymmetricTensor), (func, el) in ((zeros, zero), (ones, one))
        TensorType == SymmetricTensor && isodd(order) && continue
        isodd(order) && func == ones && continue # one not supported for Vec's
        N = Tensors.n_components(TensorType{order, dim})
        tens_arr1 = func(TensorType{order, dim}, 1)
        tens_arr2 = func(TensorType{order, dim, T}, 2, 2)
        tens_arr3 = func(TensorType{order, dim, T, N}, 3, 3, 3)
        @test tens_arr1[1] == tens_arr2[1, 1] == tens_arr3[1, 1, 1] == el(TensorType{order, dim, T})
        @test eltype(tens_arr1) == TensorType{order, dim, Float64, N}
        @test eltype(tens_arr2) == eltype(tens_arr3) == TensorType{order, dim, T, N}
    end
end
for dim in (1, 2, 3)
    # Heterogeneous tuple/type unstable function
    z(i, jkl...) = i % 2 == 0 ? 0 : float(0)
    @test Vec{dim}(ntuple(z, dim))::Vec{dim,Float64} ==
          Vec{dim}(z)::Vec{dim,Float64}
    @test Vec{dim,Float32}(ntuple(z, dim))::Vec{dim,Float32} ==
          Vec{dim,Float32}(z)::Vec{dim,Float32}
    for order in (1, 2, 3, 4)
        N = Tensors.n_components(Tensor{order,dim})
        @test Tensor{order,dim}(ntuple(z, N))::Tensor{order,dim,Float64} ==
              Tensor{order,dim}(z)::Tensor{order,dim,Float64}
        @test Tensor{order,dim,Float32}(ntuple(z, N))::Tensor{order,dim,Float32} ==
              Tensor{order,dim,Float32}(z)::Tensor{order,dim,Float32}
        @test_throws MethodError Tensor{order,dim}(ntuple(z, N+1))
        isodd(order) && continue
        N = Tensors.n_components(SymmetricTensor{order,dim})
        @test SymmetricTensor{order,dim}(ntuple(z, N))::SymmetricTensor{order,dim,Float64} ==
              SymmetricTensor{order,dim}(z)::SymmetricTensor{order,dim,Float64}
        @test SymmetricTensor{order,dim,Float32}(ntuple(z, N))::SymmetricTensor{order,dim,Float32} ==
              SymmetricTensor{order,dim,Float32}(z)::SymmetricTensor{order,dim,Float32}
        @test_throws MethodError SymmetricTensor{order,dim}(ntuple(z, N+1))
    end
end
# Number type which is not <: Real but <: Number (Tensors#154)
@test Vec{3, NotReal}((1, 2, 3)) isa Vec{3, NotReal}
end # of testset

@testset "diagm, one" begin
for T in (Float32, Float64), dim in (1,2,3)
    # diagm
    v = rand(T, dim)
    vt = (v...,)

    @test (@inferred diagm(Tensor{2, dim}, v))::Tensor{2, dim, T} == diagm(0 => v)
    @test (@inferred diagm(Tensor{2, dim}, vt))::Tensor{2, dim, T} == diagm(0 => v)
    @test (@inferred diagm(SymmetricTensor{2, dim}, v))::SymmetricTensor{2, dim, T} == diagm(0 => v)
    @test (@inferred diagm(SymmetricTensor{2, dim}, vt))::SymmetricTensor{2, dim, T} == diagm(0 => v)

    v = rand(T); vv = v * ones(T, dim)
    @test (@inferred diagm(Tensor{2, dim}, v))::Tensor{2, dim, T} == diagm(0 => vv)
    @test (@inferred diagm(SymmetricTensor{2, dim}, v))::SymmetricTensor{2, dim, T} == diagm(0 => vv)

    # one
    @test one(Tensor{2, dim, T}) == diagm(Tensor{2, dim}, one(T)) == Matrix(I, dim, dim)
    @test one(SymmetricTensor{2, dim, T}) == diagm(SymmetricTensor{2, dim}, one(T)) == Matrix(I, dim, dim)

    M = 1 # dummy
    @test one(Tensor{2, dim, T, M}) == one(Tensor{2, dim, T})
    @test one(SymmetricTensor{2, dim, T, M}) == one(SymmetricTensor{2, dim, T})

    _I =  (@inferred one(Tensor{2, dim, T}))::Tensor{2, dim, T}
    II = (@inferred one(Tensor{4, dim, T}))::Tensor{4, dim, T}
    I_sym =  (@inferred one(SymmetricTensor{2, dim, T}))::SymmetricTensor{2, dim, T}
    II_sym = (@inferred one(SymmetricTensor{4, dim, T}))::SymmetricTensor{4, dim, T}
    for i in 1:dim, j in 1:dim
        if i == j
            @test _I[i,j] == T(1)
            @test I_sym[i,j] == T(1)
        else
            @test _I[i,j] == T(0)
            @test I_sym[i,j] == T(0)
        end
        for k in 1:dim, l in 1:dim
            if i == k && j == l
                @test II[i,j,k,l] == T(1)
                if i == l && j == k
                    @test II_sym[i,j,k,l] == T(1)
                else
                    @test II_sym[i,j,k,l] == T(1) / 2
                end
            else
                @test II[i,j,k,l] == T(0)
                if i == l && j == k
                    @test II_sym[i,j,k,l] == T(1) / 2
                else
                    @test II_sym[i,j,k,l] == T(0)
                end
            end
        end
    end
end
end # of testset

@testset "base vectors" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
    eᵢ_func(i) = Tensor{1, dim, T}(j->j==i ? one(T) : zero(T))

    a = rand(Vec{dim, T})
    for i in 1:dim
        @test eᵢ(a, i) == eᵢ(typeof(a), i) == eᵢ(a)[i] == eᵢ(typeof(a))[i] == eᵢ_func(i)
    end

    b = zero(a)
    for i in 1:dim
        @test a[i] == eᵢ(a, i) ⋅ a
        b += eᵢ(a, i) * a[i]
    end
    @test a ≈ b
end
end # of testset

@testset "simple math" begin
for T in (Float32, Float64), dim in (1,2,3), order in (1,2,4), TensorType in (Tensor, SymmetricTensor)
    TensorType == SymmetricTensor && isodd(order) && continue
    t = rand(TensorType{order, dim, T})

    # Binary tensor tensor: +, -
    @test (@inferred t + t)::TensorType{order, dim} == Array(t) + Array(t)
    @test (@inferred 2*t)::TensorType{order, dim} == 2 * Array(t)
    @test (@inferred t - t)::TensorType{order, dim} == Array(t) - Array(t)
    @test (@inferred 0*t)::TensorType{order, dim} == 0 * Array(t)

    # Binary tensor number: *, /
    @test (@inferred 0.5 * t)::TensorType{order, dim} ≈ 0.5 * Array(t)
    @test (@inferred t * 0.5)::TensorType{order, dim} ≈ Array(t) * 0.5
    @test (@inferred t / 2.0)::TensorType{order, dim} ≈ Array(t) / 2.0

    # Unary: +, -
    @test (@inferred +t)::TensorType{order, dim} == zero(t) + t
    @test (@inferred -t)::TensorType{order, dim} == zero(t) - t

    if order == 2
        # Power by literal integer
        fm3, fm2, fm1, f0, fp1, fp2, fp3 = t -> t^-3, t -> t^-2, t -> t^-1, t -> t^0, t -> t^1, t -> t^2, t -> t^3
        @test (@inferred fm3(t))::typeof(t) ≈ inv(t) ⋅ inv(t) ⋅ inv(t)
        @test (@inferred fm2(t))::typeof(t) ≈ inv(t) ⋅ inv(t)
        @test (@inferred fm1(t))::typeof(t) ≈ inv(t)
        @test (@inferred f0(t))::typeof(t)  ≈ one(t)
        @test (@inferred fp1(t))::typeof(t) ≈ t
        @test (@inferred fp2(t))::typeof(t) ≈ t ⋅ t
        @test (@inferred fp3(t))::typeof(t) ≈ t ⋅ t ⋅ t
    end

    @test iszero(zero(TensorType{order,dim,T}))
    @test !iszero(ones(TensorType{order,dim,T}))
end
end # of testset

