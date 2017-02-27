@testset "basic constructors: rand, zero, ones" begin
for T in (Float32, Float64), dim in (1,2,3), order in (1,2,4)
    for op in (:rand, :zero, :ones)
        # Tensor, SymmetricTensor
        for TensorType in (Tensor, SymmetricTensor)
            TensorType == SymmetricTensor && order == 1 && continue
            @eval begin
                N = Tensors.n_components($TensorType{$order, $dim})

                t = @inferred $(op)($TensorType{$order, $dim})
                @test isa(t, $TensorType{$order, $dim, Float64})

                t = @inferred $(op)($TensorType{$order, $dim, $T})
                @test isa(t, $TensorType{$order, $dim, $T})

                t = @inferred $(op)($TensorType{$order, $dim, $T, N})
                @test isa(t, $TensorType{$order, $dim, $T})

                t = @inferred $(op)(t)
                @test isa(t, $TensorType{$order, $dim, $T})

                $op == zero && @test zero($TensorType{$order, $dim, $T}) == zeros($T, size(t))
                $op == ones && @test ones($TensorType{$order, $dim, $T}) == ones($T, size(t))
            end
        end
        # Vec
        if order == 1
            @eval begin
                t = @inferred $(op)(Vec{$dim})
                @test isa(t, Tensor{$order, $dim, Float64})
                @test isa(t, Vec{$dim, Float64})

                t = @inferred $(op)(Vec{$dim, $T})
                @test isa(t, Tensor{$order, $dim, $T})
                @test isa(t, Vec{$dim, $T})
            end
        end
    end
    for TensorType in (Tensor, SymmetricTensor), (func, el) in ((:zeros, :zero), (:ones, :one))
        TensorType == SymmetricTensor && order == 1 && continue
        order == 1 && func == :ones && continue # one not supported for Vec's
        @eval begin
            N = Tensors.n_components($TensorType{$order, $dim})
            tens_arr1 = $func($TensorType{$order, $dim}, 1)
            tens_arr2 = $func($TensorType{$order, $dim, $T}, 2, 2)
            tens_arr3 = $func($TensorType{$order, $dim, $T, N}, 3, 3, 3)
            @test tens_arr1[1] == tens_arr2[1, 1] == tens_arr3[1, 1, 1] == $el($TensorType{$order, $dim, $T})
            @test eltype(tens_arr1) == $TensorType{$order, $dim, Float64, N}
            @test eltype(tens_arr2) == eltype(tens_arr3) == $TensorType{$order, $dim, $T, N}
        end
    end
end
end # of testset

@testset "diagm, one" begin
for T in (Float32, Float64), dim in (1,2,3)
    # diagm
    v = rand(T, dim)
    vt = (v...)

    @test diagm(Tensor{2, dim}, v) == diagm(Tensor{2, dim}, vt) == diagm(v)
    @test isa(diagm(Tensor{2, dim}, v), Tensor{2, dim, T})
    @test isa(diagm(Tensor{2, dim}, vt), Tensor{2, dim, T})
    @test diagm(SymmetricTensor{2, dim}, v) == diagm(SymmetricTensor{2, dim}, vt) == diagm(v)
    @test isa(diagm(SymmetricTensor{2, dim}, v), SymmetricTensor{2, dim, T})
    @test isa(diagm(SymmetricTensor{2, dim}, vt), SymmetricTensor{2, dim, T})

    v = rand(T); vv = v * ones(T, dim)
    @test diagm(Tensor{2, dim}, v) == diagm(vv)
    @test isa(diagm(Tensor{2, dim}, v), Tensor{2, dim, T})
    @test diagm(SymmetricTensor{2, dim}, v) == diagm(vv)
    @test isa(diagm(SymmetricTensor{2, dim}, v), SymmetricTensor{2, dim, T})

    # one
    @test one(Tensor{2, dim, T}) == diagm(Tensor{2, dim}, one(T)) == eye(T, dim, dim)
    @test one(SymmetricTensor{2, dim, T}) == diagm(SymmetricTensor{2, dim}, one(T)) == eye(T, dim, dim)

    M = 1 # dummy
    @test one(Tensor{2, dim, T, M}) == one(Tensor{2, dim, T})
    @test one(SymmetricTensor{2, dim, T, M}) == one(SymmetricTensor{2, dim, T})

    I = one(Tensor{2, dim, T})
    I_sym = one(SymmetricTensor{2, dim, T})
    II = one(Tensor{4, dim, T})
    II_sym = one(SymmetricTensor{4, dim, T})
    for i in 1:dim, j in 1:dim
        if i == j
            @test I[i,j] == T(1)
            @test I_sym[i,j] == T(1)
        else
            @test I[i,j] == T(0)
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
for T in (Float32, Float64), dim in (1,2,3)
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
    TensorType == SymmetricTensor && order == 1 && continue
    @eval begin
        t = rand($TensorType{$order, $dim, $T})

        # Binary tensor tensor: +, -
        @test (@inferred t + t) == 2 * t == 2 * Array(t)
        @test isa(t + t, $TensorType{$order, $dim})

        @test (@inferred t - t) == zero(t) == 0 * Array(t)
        @test isa(t - t, $TensorType{$order, $dim})

        # Binary tensor number: *, /
        @test 0.5 * t ≈ t * 0.5 ≈ t / 2.0 ≈ 0.5 * Array(t)
        @test isa(0.5 * t, $TensorType{$order, $dim})
        @test isa(t * 0.5, $TensorType{$order, $dim})
        @test isa(t / 2.0, $TensorType{$order, $dim})

        @test (@inferred rand(t) * 0.0) == zero(t)

        # Unary: +, -
        @test (@inferred +t) == zero(t) + t
        @test isa(+t, $TensorType{$order, $dim})

        @test (@inferred -t) == zero(t) - t
        @test isa(-t, $TensorType{$order, $dim})
    end
end
end # of testset

@testset "create with a function" begin
for T in (Float32, Float64)
    for dim in (1,2,3)
        fi = (i) -> cos(i)
        fij = (i,j) -> cos(i) + sin(j)
        fijkl = (i, j, k ,l) -> cos(i) + sin(j) + tan(k) + exp(l)

        vf = Vec{dim, T}(fi)
        af = Tensor{1, dim, T}(fi)
        Af = Tensor{2, dim, T}(fij)
        AAf = Tensor{4, dim, T}(fijkl)
        Af_sym = SymmetricTensor{2, dim, T}(fij)
        AAf_sym = SymmetricTensor{4, dim, T}(fijkl)

        # Make sure we get the specified eltype
        @test isa(vf, Tensor{1, dim, T})
        @test isa(af, Tensor{1, dim, T})
        @test isa(Af, Tensor{2, dim, T})
        @test isa(AAf, Tensor{4, dim, T})
        @test isa(Af_sym, SymmetricTensor{2, dim, T})
        @test isa(AAf_sym, SymmetricTensor{4, dim, T})

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

@testset "create from Array" begin
for (T1, T2) in ((Float32, Float64), (Float64, Float32)), order in (1,2,4), dim in (1,2,3)
    At = rand(Tensor{order, dim})
    gen_data = rand(T1, size(At))

    @test Tensor{order, dim}(gen_data) ≈ gen_data
    @test Tensor{order, dim, T2}(gen_data) ≈ gen_data
    @test isa(Tensor{order, dim}(gen_data), Tensor{order, dim, T1})
    @test isa(Tensor{order, dim, T2}(gen_data), Tensor{order, dim, T2})

    if order != 1
        As = rand(SymmetricTensor{order, dim})
        gen_data = rand(T1, size(As))
        At = Tensor{order, dim, T1}(gen_data)
        Ats = symmetric(At)
        gen_sym_data_full = Array(Ats)
        gen_sym_data = T1[Ats.data[i] for i in 1:length(Ats.data)]

        @test SymmetricTensor{order, dim}(gen_sym_data_full) ≈ Ats
        @test SymmetricTensor{order, dim}(gen_sym_data) ≈ Ats
        @test SymmetricTensor{order, dim, T2}(gen_sym_data_full) ≈ Ats
        @test SymmetricTensor{order, dim, T2}(gen_sym_data) ≈ Ats

        @test isa(SymmetricTensor{order, dim}(gen_sym_data_full), SymmetricTensor{order, dim, T1})
        @test isa(SymmetricTensor{order, dim}(gen_sym_data), SymmetricTensor{order, dim, T1})
        @test isa(SymmetricTensor{order, dim, T2}(gen_sym_data_full), SymmetricTensor{order, dim, T2})
        @test isa(SymmetricTensor{order, dim, T2}(gen_sym_data), SymmetricTensor{order, dim, T2})
    end
end
end # of testset

@testset "indexing" begin
for T in (Float32, Float64), dim in (1,2,3), order in (1,2,4)
    if order == 1
        data = rand(T, dim)
        vec = Tensor{order, dim, T}(data)
        for i in 1:dim+1
            if i > dim
                @test_throws BoundsError vec[i]
            else
                @test vec[i] ≈ data[i]
            end
        end
        @test vec[:] == vec
        @test typeof(vec[:]) <: Vec{dim, T}
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
                @test S[i,:] ≈ data[i,:]
                @test typeof(S[i,:]) <: Tensor{1, dim, T}
                @test S[:,j] ≈ data[:,j]
                @test typeof(S[:,j]) <: Tensor{1, dim, T}
                @test Ssym[i,:] ≈ symdata[i,:]
                @test typeof(Ssym[i,:]) <: Tensor{1, dim, T}
                @test Ssym[:,j] ≈ symdata[:,j]
                @test typeof(Ssym[:,j]) <: Tensor{1, dim, T}
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

@testset "norm, trace, det, inv, eig" begin
for T in (Float32, Float64), dim in (1,2,3)
    # norm
    for order in (1,2,4)
        t = rand(Tensor{order, dim, T})
        @test norm(t) ≈ sqrt(sum(abs2, Array(t)[:]))
        @test isa(norm(t), T)
        if order != 1
            t_sym = rand(SymmetricTensor{order, dim, T})
            @test norm(t_sym) ≈ sqrt(sum(abs2, Array(t_sym)[:]))
            @test isa(norm(t_sym), T)
        end
    end

    # trace, dev, det, inv (only for second order tensors)
    t = rand(Tensor{2, dim, T})
    t_sym = rand(SymmetricTensor{2, dim, T})

    @test trace(t) == sum([t[i,i] for i in 1:dim])
    @test trace(t_sym) == sum([t_sym[i,i] for i in 1:dim])

    @test trace(t) ≈ vol(t) ≈ mean(t)*3.0
    @test trace(t_sym) ≈ vol(t_sym) ≈ mean(t_sym)*3.0

    @test dev(t) ≈ Array(t) - 1/3*trace(t)*eye(dim)
    @test isa(dev(t), Tensor{2, dim, T})
    @test dev(t_sym) ≈ Array(t_sym) - 1/3*trace(t_sym)*eye(dim)
    @test isa(dev(t_sym), SymmetricTensor{2, dim, T})

    @test det(t) ≈ det(Array(t))
    @test isa(det(t), T)
    @test det(t_sym) ≈ det(Array(t_sym))
    @test isa(det(t_sym), T)

    @test inv(t) ≈ inv(Array(t))
    @test isa(inv(t), Tensor{2, dim, T})
    @test inv(t_sym) ≈ inv(Array(t_sym))
    @test isa(inv(t_sym), SymmetricTensor{2, dim, T})

    Λ, Φ = eig(t_sym)
    Λa, Φa = eig(Array(t_sym))

    @test Λ ≈ Λa
    for i in 1:dim
        # scale with first element of eigenvector to account for possible directions
        @test Φ[:, i]*Φ[1, i] ≈ Φa[:, i]*Φa[1, i]
    end
end
end # of testset

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
@testset "tensor identities" begin
for T in (Float32, Float64)
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
        @test (A ⋅ a) ⋅ (B ⋅ b) ≈ (A.' ⋅ B) ⊡ (a ⊗ b)
        @test (A ⋅ a) ⊗ b ≈ A ⋅ (a ⊗ b)
        @test a ⊗ (A ⋅ b) ≈ (A ⋅ (b ⊗ a)).'
        @test a ⊗ (A ⋅ b) ≈ (a ⊗ b) ⋅ A.'

        @test A ⊡ I ≈ trace(A)
        @test det(A) ≈ det(A.')
        @test trace(inv(A) ⋅ A) ≈ dim
        @test inv(A) ⋅ A ≈ I

        @test (I ⊗ I) ⊡ A ≈ trace(A) * I
        @test (I ⊗ I) ⊡ A ⊡ A ≈ trace(A)^2

        @test A ⋅ a ≈ a ⋅ A'

        A_sym = rand(SymmetricTensor{2, dim})
        B_sym = rand(SymmetricTensor{2, dim})
        C_sym = rand(SymmetricTensor{2, dim})
        I_sym = one(SymmetricTensor{2, dim})

        @test A_sym ⊡ I_sym ≈ trace(A_sym)
        @test det(A_sym) ≈ det(A_sym.')

        @test (I_sym ⊗ I_sym) ⊡ A_sym ≈ trace(A_sym) * I_sym
        @test ((I_sym ⊗ I_sym) ⊡ A_sym) ⊡ A_sym ≈ trace(A_sym)^2
    end
end

for T in (Float32, Float64)
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
        @test II ⊡ A ⊡ A ≈ (trace(A' ⋅ A))

        @test II ⊡ AA_sym ≈ AA_sym
        @test AA_sym ⊡ II ≈ AA_sym
        @test II ⊡ A_sym ≈ A_sym
        @test A_sym ⊡ II ≈ A_sym
        @test II ⊡ A_sym ⊡ A_sym ≈ (trace(A_sym' ⋅ A_sym))

        @test II_sym ⊡ AA_sym ≈ AA_sym
        @test AA_sym ⊡ II_sym ≈ AA_sym
        @test II_sym ⊡ A_sym ≈ A_sym
        @test A_sym ⊡ II_sym ≈ A_sym
        @test II_sym ⊡ A_sym ⊡ A_sym ≈ (trace(A_sym' ⋅ A_sym))
    end
end
end # of testset

@testset "promotion/conversion" begin
const T = Float32
const WIDE_T = widen(T)
for dim in (1,2,3), order in (1,2,4)

    tens = Tensor{order, dim, T, dim^order}
    tens_wide = Tensor{order, dim, WIDE_T, dim^order}

    @test promote_type(tens, tens) == tens
    @test promote_type(tens_wide, tens) == tens_wide
    @test promote_type(tens, tens_wide) == tens_wide
    @test promote_type(tens_wide, tens_wide) == tens_wide

    At = rand(tens)
    Bt = rand(tens_wide)
    @test isa(At + Bt, tens_wide)
    @test isa(convert(Tensor, At), tens)
    @test isa(convert(Tensor{order, dim, T}, At), tens)
    @test isa(convert(tens, At), tens)
    @test isa(convert(Tensor{order, dim, WIDE_T}, At), tens_wide)
    @test isa(convert(tens_wide, At), tens_wide)

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
        @test isa(As + Bs, sym_wide)

        @test isa(convert(SymmetricTensor, As), sym)
        @test isa(convert(SymmetricTensor{order, dim, T}, As), sym)
        @test isa(convert(sym, As), sym)
        @test isa(convert(SymmetricTensor{order, dim, WIDE_T}, As), sym_wide)
        @test isa(convert(sym_wide, As), sym_wide)

        # SymmetricTensor -> Tensor
        @test convert(Tensor, As) ≈ As ≈ Array(As)
        @test convert(Tensor{order, dim}, As) ≈ As ≈ Array(As)
        @test convert(Tensor{order, dim, WIDE_T}, As) ≈ As ≈ Array(As)
        @test convert(tens_wide, As) ≈ As ≈ Array(As)

        @test isa(convert(Tensor, As), tens)
        @test isa(convert(Tensor{order, dim}, As), tens)
        @test isa(convert(Tensor{order, dim, WIDE_T}, As), tens_wide)
        @test isa(convert(tens_wide, As), tens_wide)

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

@testset "exceptions" begin
    # normal multiplication
    A = rand(Tensor{2, 3})
    B = rand(Tensor{2, 3})
    @test_throws Exception A*B
    @test_throws Exception A'*B
    @test_throws Exception A.'*B
    @test_throws Exception A\B

    # issue 75
    @test_throws MethodError A+1
    @test_throws MethodError 1+A
    @test_throws MethodError A-1
    @test_throws MethodError 1-A
end # of testset
