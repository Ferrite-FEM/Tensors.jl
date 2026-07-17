@testset "construct from function" begin
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

        # Test fully specified constructors
        for vec in (vf,af)
            @test vec == typeof(vec)(i->vec[i])
        end
        for t2 in (Af, Af_sym)
            @test t2 == typeof(t2)((i,j)->t2[i,j])
        end
        for t4 in (AAf, AAf_sym)
            @test t4 == typeof(t4)((i,j,k,l) -> t4[i,j,k,l])
        end 
    end
end
end # of testset

@testset "construct from Array" begin
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

@testset "indexing" begin
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

@testset "tensor identities" begin
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

for T in (Float64,)
    for dim in 1:3
        # Tested operations
        # S2 ⋅ S3,  S3 ⋅ S2 
        # S1 ⊗ S2, S2 ⊗ S1
        # S3 ⊡ S2, S2 ⊡ S3
        
        # Identities involving 3rd order tensors 
        I2 = one(Tensor{2,dim,T})
        A2 = rand(Tensor{2,dim,T})
        S2 = rand(Tensor{2,dim,T})
        S2s = rand(SymmetricTensor{2,dim,T})
        S2sr = Tensor{2,dim,T}(S2s)
        v = rand(Vec{dim,T})
        u = rand(Vec{dim,T})
        @test (u⊗I2)⋅v ≈ (u⊗v)
        @test u⋅(I2⊗v) ≈ (u⊗v)
        @test I2⋅(v⊗S2) ≈ (v⊗S2)
        @test v ⊗ (S2⋅A2) ≈ (v ⊗ S2)⋅A2
        @test v ⊗ S2s ≈ v ⊗ S2sr
        @test S2s ⊗ v ≈ S2sr ⊗ v
        @test (v⊗S2)⊡A2 ≈ v*(S2⊡A2)
        @test (v⊗S2)⊡S2s ≈ v*(S2⊡S2s)
        @test A2⊡(S2⊗v) ≈ v*(S2⊡A2)
        @test S2s⊡(S2⊗v) ≈ v*(S2⊡S2s)
    end
end
end # of testset

@testset "promotion/conversion" begin
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

using Unitful
@testset "unitful" begin; for dim in 1:3
        A = rand(Tensor{2,dim})
        As = rand(SymmetricTensor{2,dim})
        @test one(A * 1u"Pa")::Tensor{2,dim,Float64} == one(A)::Tensor{2,dim,Float64}
        @test one(As * 1u"Pa")::SymmetricTensor{2,dim,Float64} == one(As)::SymmetricTensor{2,dim,Float64}

        # Eigen of 2nd order
        M = rand(dim, dim) + 5I; M = M'M
        S = SymmetricTensor{2,dim}(M)
        SU = SymmetricTensor{2,dim}(M * 1u"Pa")
        ST = convert(Tensor, S)
        SUT = convert(Tensor, SU)
        @test eigvals(Hermitian(M)) ≈ eigvals(S) ≈ ustrip.(eigvals(SU))
        same_no_sign(args...) = all(args) do arg; all(1:dim) do i
            args[1][:, i] ≈ arg[:, i] || args[1][:, i] ≈ -arg[:, i]
        end end
        @test same_no_sign(eigvecs(Hermitian(M)), eigvecs(S), ustrip.(eigvecs(SU)))
        @test eltype(eigvals(SU)) == typeof(1.0u"Pa")
        @test eltype(eigvecs(SU)) == typeof(one(1.0u"Pa"))
        # inv
        @test inv(M) ≈ inv(S) ≈ inv(ST) ≈
              ustrip.(inv(SU)) ≈ ustrip.(inv(SUT))
        # (to|from)(voigt|mandel)
        @test fromvoigt(SymmetricTensor{2,dim}, tovoigt(SU)) == SU
        @test fromvoigt(SymmetricTensor{2,dim}, tovoigt(SU; offdiagscale=5.0); offdiagscale=5.0) ≈ SU
        @test frommandel(SymmetricTensor{2,dim}, tomandel(SU)) ≈ SU
        @test fromvoigt(Tensor{2,dim}, tovoigt(SUT)) == SUT
        # Eigen of 4th order
        M = rand(dim+dim, dim+dim) + 5I; M = M'M
        S = fromvoigt(SymmetricTensor{4,dim}, M)
        SU = fromvoigt(SymmetricTensor{4,dim}, M * 1u"Pa")
        @test eigvals(S) ≈ ustrip.(eigvals(SU))
        @test eigvecs(S) ≈ eigvecs(SU)
        @test eltype(eigvals(SU)) == typeof(1.0u"Pa")
        @test eltype(eigvecs(SU)[1]) == typeof(one(1.0u"Pa"))
        # inv
        @test inv(S) ⊡ S ≈ inv(SU) ⊡ SU ≈ one(S)
        M = rand(dim^2, dim^2) + 5I; M = M'M
        ST = fromvoigt(Tensor{4,dim}, M)
        SUT = fromvoigt(Tensor{4,dim}, M * 1u"Pa")
        @test inv(ST) ⊡ ST ≈ inv(SUT) ⊡ SUT ≈ one(ST)
        # (to|from)(voigt|mandel)
        @test fromvoigt(SymmetricTensor{4,dim}, tovoigt(SU)) == SU
        @test fromvoigt(SymmetricTensor{4,dim}, tovoigt(SU; offdiagscale=5.0); offdiagscale=5.0) ≈ SU
        @test frommandel(SymmetricTensor{4,dim}, tomandel(SU)) ≈ SU
        @test fromvoigt(Tensor{4,dim}, tovoigt(SUT)) == SUT
end end

@testset "exceptions" begin
    # normal multiplication
    A = rand(Tensor{2, 3})
    B = rand(Tensor{2, 3})
    @test_throws Exception A*B
    @test_throws Exception A'*B

    AA = rand(Tensor{4, 2})
    A2 = rand(Tensor{2, 2})
    @test_throws DimensionMismatch A + A2
    @test_throws DimensionMismatch AA - A

    # transpose/adjoint of Vec
    x = rand(Vec{3})
    @test_throws ArgumentError x'
    @test_throws ArgumentError transpose(x)
    @test_throws ArgumentError adjoint(x)

    # Non-implemented single contractions
    t10 = Tensor{10, 1, Float64, 10}(tuple((1.0 for _ in 1:10)...));
    @test_throws ArgumentError (t10 ⋅ t10)
end # of testset
