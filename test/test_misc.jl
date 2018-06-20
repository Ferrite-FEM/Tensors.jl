@testsection "constructors" begin
for T in (Float32, Float64, F64), dim in (1,2,3), order in (1,2,4)
    for op in (rand, zero, ones, randn)
        # Tensor, SymmetricTensor
        for TensorType in (Tensor, SymmetricTensor)
            TensorType == SymmetricTensor && order == 1 && continue
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
    for TensorType in (Tensor, SymmetricTensor), (func, el) in ((zeros, zero), (ones, one))
        TensorType == SymmetricTensor && order == 1 && continue
        order == 1 && func == ones && continue # one not supported for Vec's
        N = Tensors.n_components(TensorType{order, dim})
        tens_arr1 = func(TensorType{order, dim}, 1)
        tens_arr2 = func(TensorType{order, dim, T}, 2, 2)
        tens_arr3 = func(TensorType{order, dim, T, N}, 3, 3, 3)
        @test tens_arr1[1] == tens_arr2[1, 1] == tens_arr3[1, 1, 1] == el(TensorType{order, dim, T})
        @test eltype(tens_arr1) == TensorType{order, dim, Float64, N}
        @test eltype(tens_arr2) == eltype(tens_arr3) == TensorType{order, dim, T, N}
    end
end
end # of testset

@testsection "diagm, one" begin
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

@testsection "base vectors" begin
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

@testsection "simple math" begin
for T in (Float32, Float64), dim in (1,2,3), order in (1,2,4), TensorType in (Tensor, SymmetricTensor)
    TensorType == SymmetricTensor && order == 1 && continue
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
end
end # of testset

@testsection "constrct func" begin
for T in (Float32, Float64, F64)
    for dim in (1,2,3)
        fi = (i) -> cos(i)
        fij = (i,j) -> cos(i) + sin(j)
        fijkl = (i, j, k ,l) -> cos(i) + sin(j) + tan(k) + exp(l)

        vf = (@inferred Vec{dim, T}(fi))::Tensor{1, dim, T}
        af = (@inferred Tensor{1, dim, T}(fi))::Tensor{1, dim, T}
        Af = (@inferred Tensor{2, dim, T}(fij))::Tensor{2, dim, T}
        AAf = (@inferred Tensor{4, dim, T}(fijkl))::Tensor{4, dim, T}
        Af_sym = (@inferred SymmetricTensor{2, dim, T}(fij))::SymmetricTensor{2, dim, T}
        AAf_sym = (@inferred SymmetricTensor{4, dim, T}(fijkl))::SymmetricTensor{4, dim, T}

        for i in 1:dim
            @test vf[i] == T(fi(i))
            @test af[i] == T(fi(i))
            for j in 1:dim
                @test Af[i,j] == T(fij(i, j))
                for k in 1:dim, l in 1:dim
                    @test AAf[i,j,k,l] == T(fijkl(i,j,k,l))
                end
            end
        end

        for j in 1:dim, i in j:dim
            @test Af_sym[i,j] == T(fij(i, j))
            for l in 1:dim, k in l:dim
                 @test AAf_sym[i,j,k,l] == T(fijkl(i,j,k,l))
            end
        end
    end
end
end # of testset

@testsection "constrct Arr" begin
for (T1, T2) in ((Float32, Float64), (Float64, Float32)), order in (1,2,4), dim in (1,2,3)
    At = rand(Tensor{order, dim})
    gen_data = rand(T1, size(At))

    @test (@inferred Tensor{order, dim}(gen_data))::Tensor{order, dim, T1}     ≈ gen_data
    @test (@inferred Tensor{order, dim, T2}(gen_data))::Tensor{order, dim, T2} ≈ gen_data

    if order != 1
        As = rand(SymmetricTensor{order, dim})
        gen_data = rand(T1, size(As))
        At = Tensor{order, dim, T1}(gen_data)
        Ats = symmetric(At)
        gen_sym_data_full = Array(Ats)
        gen_sym_data = T1[Ats.data[i] for i in 1:length(Ats.data)]

        @test (@inferred SymmetricTensor{order, dim}(gen_sym_data_full))::SymmetricTensor{order, dim, T1}     ≈ Ats
        @test (@inferred SymmetricTensor{order, dim}(gen_sym_data))::SymmetricTensor{order, dim, T1}          ≈ Ats
        @test (@inferred SymmetricTensor{order, dim, T2}(gen_sym_data_full))::SymmetricTensor{order, dim, T2} ≈ Ats
        @test (@inferred SymmetricTensor{order, dim, T2}(gen_sym_data))::SymmetricTensor{order, dim, T2}      ≈ Ats
    end
end
end # of testset

@testsection "indexing" begin
for T in (Float32, Float64, F64), dim in (1,2,3), order in (1,2,4)
    if order == 1
        data = rand(T, dim)
        vect = Tensor{order, dim, T}(data)
        for i in 1:dim+1
            if i > dim
                @test_throws BoundsError vect[i]
            else
                @test vect[i] ≈ data[i]
            end
        end
        @test (@inferred vect[:])::Vec{dim, T} == vect
    elseif order == 2
        data = rand(T, dim, dim)
        symdata = data + data'
        S = Tensor{order,dim, T}(data)
        Ssym = SymmetricTensor{order,dim, T}(symdata)
        @test_throws ArgumentError S[:]
        @test_throws ArgumentError Ssym[:]
        for i in 1:dim+1, j in 1:dim+1
            if i > dim || j > dim
                @test_throws BoundsError S[i, j]
                @test_throws BoundsError Ssym[i, j]
            else
                @test S[i, j] ≈ data[i, j]
                @test Ssym[i, j] ≈ symdata[i, j]
                # Slice
                @test (@inferred S[i,:])::Tensor{1, dim, T} ≈ data[i,:]
                @test (@inferred S[:,j])::Tensor{1, dim, T} ≈ data[:,j]
                @test (@inferred Ssym[i,:])::Tensor{1, dim, T} ≈ symdata[i,:]
                @test (@inferred Ssym[:,j])::Tensor{1, dim, T} ≈ symdata[:,j]
            end
        end
    elseif order == 4
        data = rand(T,dim,dim,dim,dim)
        S = Tensor{order,dim, T}(data)
        Ssym = symmetric(S)
        symdata = Array(Ssym)
        @test_throws ArgumentError S[:]
        @test_throws ArgumentError Ssym[:]
        for i in 1:dim+1, j in 1:dim+1, k in 1:dim+1, l in 1:dim+1
            if i > dim || j > dim || k > dim || l > dim
                @test_throws BoundsError S[i, j, k, l]
                @test_throws BoundsError Ssym[i, j, k, l]
            else
                @test S[i, j, k, l] ≈ data[i, j, k, l]
                @test Ssym[i, j, k, l] ≈ symdata[i, j, k, l]
            end
        end
    end
end
end # of testset

@testsection "norm, trace, det, inv, eig" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
    # norm
    for order in (1,2,4)
        t = rand(Tensor{order, dim, T})
        @test (@inferred norm(t))::T ≈ sqrt(sum(abs2, Array(t)[:]))
        if order != 1
            t_sym = rand(SymmetricTensor{order, dim, T})
            @test (@inferred norm(t_sym))::T ≈ sqrt(sum(abs2, Array(t_sym)[:]))
        end
    end

    # trace, vol, dev, det, inv (only for second order tensors)
    t = rand(Tensor{2, dim, T})
    t_sym = rand(SymmetricTensor{2, dim, T})

    @test (@inferred tr(t))::T == sum([t[i,i] for i in 1:dim])
    @test (@inferred tr(t_sym))::T == sum([t_sym[i,i] for i in 1:dim])

    @test tr(t) ≈ mean(t)*3.0
    @test tr(t_sym) ≈ mean(t_sym)*3.0

    @test (@inferred vol(t))::Tensor{2, dim, T} ≈ mean(t) * Matrix(I, dim, dim)
    @test (@inferred vol(t_sym))::SymmetricTensor{2, dim, T} ≈ mean(t_sym) * Matrix(I, dim, dim)

    @test (@inferred dev(t))::Tensor{2, dim, T} ≈ Array(t) - 1/3*tr(t)* Matrix(I, dim, dim)
    @test (@inferred dev(t_sym))::SymmetricTensor{2, dim, T} ≈ Array(t_sym) - 1/3*tr(t_sym)* Matrix(I, dim, dim)

    @test (@inferred det(t))::T ≈ det(Array(t))
    @test (@inferred det(t_sym))::T ≈ det(Array(t_sym))

    @test (@inferred inv(t))::Tensor{2, dim, T} ≈ inv(Array(t))
    @test (@inferred inv(t_sym))::SymmetricTensor{2, dim, T} ≈ inv(Array(t_sym))

    # inv for fourth order tensors
    AA = rand(Tensor{4, dim, T})
    AA_sym = rand(SymmetricTensor{4, dim, T})
    @test AA ⊡ (@inferred inv(AA))::Tensor{4, dim, T} ≈ one(Tensor{4, dim, T})
    @test AA_sym ⊡ (@inferred inv(AA_sym))::SymmetricTensor{4, dim, T} ≈ one(SymmetricTensor{4, dim, T})

    E = @inferred eigen(t_sym)
    Λ, Φ = E
    Λa, Φa = eigen(Array(t_sym))

    @test Λ ≈ (@inferred eigvals(t_sym)) ≈ eigvals(E) ≈ Λa
    @test Φ ≈ (@inferred eigvecs(t_sym)) ≈ eigvecs(E)
    for i in 1:dim
        # scale with first element of eigenvector to account for possible directions
        @test Φ[:, i]*Φ[1, i] ≈ Φa[:, i]*Φa[1, i]
    end

    # test eigenfactorizations for a diagonal tensor
    v = rand(T, dim)
    d_sym = diagm(SymmetricTensor{2, dim, T}, v)
    E = @inferred eigen(d_sym)
    Λ, Φ = E
    Λa, Φa = eigen(Symmetric(Array(d_sym)))

    @test Λ ≈ (@inferred eigvals(d_sym)) ≈ eigvals(E) ≈ Λa
    @test Φ ≈ (@inferred eigvecs(d_sym)) ≈ eigvecs(E)

    # sqrt
    Apd = tdot(t_sym)
    @test sqrt(Apd) ⋅ sqrt(Apd) ≈ Apd
end
end # of testset

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
@testsection "tensor identities" begin
for T in (Float32, Float64, F64)
    for dim in (1,2,3)
        # Identities with second order and first order
        A = rand(Tensor{2, dim, T})
        B = rand(Tensor{2, dim, T})
        C = rand(Tensor{2, dim, T})
        I = one(Tensor{2, dim, T})
        a = rand(Tensor{1, dim, T})
        b = rand(Tensor{1, dim, T})

        @test A ⊡ B ≈ (A' ⋅ B) ⊡ one(A)
        @test A ⊡ (a ⊗ b) ≈ (A ⋅ b) ⋅ a
        @test (A ⋅ a) ⋅ (B ⋅ b) ≈ (A' ⋅ B) ⊡ (a ⊗ b)
        @test (A ⋅ a) ⊗ b ≈ A ⋅ (a ⊗ b)
        @test a ⊗ (A ⋅ b) ≈ (A ⋅ (b ⊗ a))'
        @test a ⊗ (A ⋅ b) ≈ (a ⊗ b) ⋅ A'

        @test A ⊡ I ≈ tr(A)
        @test det(A) ≈ det(A')
        @test tr(inv(A) ⋅ A) ≈ dim
        @test inv(A) ⋅ A ≈ A \ A ≈ I
        @test inv(A) ⋅ a ≈ A \ a

        @test (I ⊗ I) ⊡ A ≈ tr(A) * I
        @test (I ⊗ I) ⊡ A ⊡ A ≈ tr(A)^2

        @test A ⋅ a ≈ a ⋅ A'

        A_sym = rand(SymmetricTensor{2, dim})
        B_sym = rand(SymmetricTensor{2, dim})
        C_sym = rand(SymmetricTensor{2, dim})
        I_sym = one(SymmetricTensor{2, dim})

        @test A_sym ⊡ I_sym ≈ tr(A_sym)
        @test det(A_sym) ≈ det(A_sym')

        @test (I_sym ⊗ I_sym) ⊡ A_sym ≈ tr(A_sym) * I_sym
        @test ((I_sym ⊗ I_sym) ⊡ A_sym) ⊡ A_sym ≈ tr(A_sym)^2
    end
end

for T in (Float32, Float64, F64)
    for dim in (1,2,3)
        # Identities with identity tensor
        II = one(Tensor{4, dim, T})
        I = one(Tensor{2, dim, T})
        AA = rand(Tensor{4, dim, T})
        A = rand(Tensor{2, dim, T})
        II_sym = one(SymmetricTensor{4, dim, T})
        I_sym = one(SymmetricTensor{2, dim, T})
        AA_sym = rand(SymmetricTensor{4, dim, T})
        A_sym = rand(SymmetricTensor{2, dim, T})

        @test II ⊡ AA ≈ AA
        @test AA ⊡ II ≈ AA
        @test II ⊡ A ≈ A
        @test A ⊡ II ≈ A
        @test II ⊡ A ⊡ A ≈ (tr(A' ⋅ A))

        @test II ⊡ AA_sym ≈ AA_sym
        @test AA_sym ⊡ II ≈ AA_sym
        @test II ⊡ A_sym ≈ A_sym
        @test A_sym ⊡ II ≈ A_sym
        @test II ⊡ A_sym ⊡ A_sym ≈ (tr(A_sym' ⋅ A_sym))

        @test II_sym ⊡ AA_sym ≈ AA_sym
        @test AA_sym ⊡ II_sym ≈ AA_sym
        @test II_sym ⊡ A_sym ≈ A_sym
        @test A_sym ⊡ II_sym ≈ A_sym
        @test II_sym ⊡ A_sym ⊡ A_sym ≈ (tr(A_sym' ⋅ A_sym))
    end
end
end # of testset

@testsection "promotion/conversion" begin
T = Float32
WIDE_T = widen(T)
for dim in (1,2,3), order in (1,2,4)

    tens = Tensor{order, dim, T, dim^order}
    tens_wide = Tensor{order, dim, WIDE_T, dim^order}

    @test promote_type(tens, tens) == tens
    @test promote_type(tens_wide, tens) == tens_wide
    @test promote_type(tens, tens_wide) == tens_wide
    @test promote_type(tens_wide, tens_wide) == tens_wide

    At = rand(tens)
    Bt = rand(tens_wide)
    @test isa((@inferred At + Bt), tens_wide)
    @test isa((@inferred convert(Tensor, At)), tens)
    @test isa((@inferred convert(Tensor{order, dim, T}, At)), tens)
    @test isa((@inferred convert(tens, At)), tens)
    @test isa((@inferred convert(Tensor{order, dim, WIDE_T}, At)), tens_wide)
    @test isa((@inferred convert(tens_wide, At)), tens_wide)

    if order != 1
        M = Tensors.n_components(SymmetricTensor{order, dim})
        sym = SymmetricTensor{order, dim, T, M}
        sym_wide = SymmetricTensor{order, dim, WIDE_T, M}

        @test promote_type(sym, sym) == sym
        @test promote_type(sym_wide, sym) == sym_wide
        @test promote_type(sym, sym_wide) == sym_wide
        @test promote_type(sym_wide, sym_wide) == sym_wide

        @test promote_type(sym, tens) == tens
        @test promote_type(tens, sym) == tens
        @test promote_type(sym_wide, tens) == tens_wide
        @test promote_type(tens, sym_wide) == tens_wide
        @test promote_type(sym, tens_wide) == tens_wide
        @test promote_type(tens_wide, sym) == tens_wide
        @test promote_type(sym_wide, tens_wide) == tens_wide

        As = rand(sym)
        Bs = rand(sym_wide)
        @test isa((@inferred As + Bs), sym_wide)

        @test isa((@inferred convert(SymmetricTensor, As)), sym)
        @test isa((@inferred convert(SymmetricTensor{order, dim, T}, As)), sym)
        @test isa((@inferred convert(sym, As)), sym)
        @test isa((@inferred convert(SymmetricTensor{order, dim, WIDE_T}, As)), sym_wide)
        @test isa((@inferred convert(sym_wide, As)), sym_wide)

        # SymmetricTensor -> Tensor
        @test (@inferred convert(Tensor, As))::tens ≈ As ≈ Array(As)
        @test (@inferred convert(Tensor{order, dim}, As))::tens ≈ As ≈ Array(As)
        @test (@inferred convert(Tensor{order, dim, WIDE_T}, As))::tens_wide ≈ As ≈ Array(As)
        @test (@inferred convert(tens_wide, As))::tens_wide ≈ As ≈ Array(As)

        # Tensor -> SymmetricTensor
        if dim != 1
            @test_throws InexactError convert(SymmetricTensor, At)
            @test_throws InexactError convert(SymmetricTensor{order, dim}, At)
            @test_throws InexactError convert(SymmetricTensor{order, dim, WIDE_T}, At)
            @test_throws InexactError convert(typeof(Bs), At)
        end
    end
end
end  # of testset

@testsection "exceptions" begin
    # normal multiplication
    A = rand(Tensor{2, 3})
    B = rand(Tensor{2, 3})
    @test_throws Exception A*B
    @test_throws Exception A'*B

    AA = rand(Tensor{4, 2})
    A2 = rand(Tensor{2, 2})
    @test_throws DimensionMismatch A + A2
    @test_throws DimensionMismatch AA - A
end # of testset
