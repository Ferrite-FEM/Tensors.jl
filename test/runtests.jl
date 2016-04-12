using ContMechTensors

if VERSION >= v"0.5-"
    using Base.Test
else
    using BaseTestNext
    const Test = BaseTestNext
end

import ContMechTensors: n_independent_components, ArgumentError, get_data

include("test_ops.jl")

@testset "Constructors and simple math ops." begin
for dim in (1,2,3)
    for order in (1,2,4)
        n_sym = n_independent_components(dim, true)
        n = n_independent_components(dim, false)


        ################
        # Constructors #
        ################

        t = @inferred rand(Tensor{order, dim})

        if order != 1
            @inferred rand(Tensor{order, dim})
        end


        ###############
        # Simple math #
        ###############

        t = rand(Tensor{order, dim})
        t_one = one(Tensor{order, dim})

        @test (@inferred t + t) == 2*t
        @test (@inferred -t) == zero(t) - t
        @test 2*t == t*2
        @test 0.5 * t ≈ t / 2.0
        @test (@inferred rand(t) * 0.0) == zero(t)

        if order != 1
            t_sym =  @inferred rand(SymmetricTensor{order, dim})
            @test (@inferred t_sym + t_sym) == 2*t_sym
            @test (@inferred -t_sym) == zero(t_sym) - t_sym
            @test (@inferred 2*t_sym) == t_sym*2

            @test 2*t_sym == t_sym*2

            @test (@inferred rand(t_sym) * 0.0) == zero(t_sym)
        end
    end
end
end # of testset


############
# Indexing #
############
@testset "Indexing" begin
for dim in (1,2,3)
    for order in (1,2,4)
        n_sym = n_independent_components(dim, true)
        n = n_independent_components(dim, false)

        if order == 1
            vec = Tensor{order, dim}(rand(dim))
            if dim == 1
                @test_throws BoundsError vec[2]
            elseif dim == 2
                @test_throws BoundsError vec[3]
            elseif dim == 3
                @test_throws BoundsError vec[4]
            else
                for t in (SymmetricTensor(rand(n), Val{dim}), Tensor(rand(n), Val{dim}))
                    if order == 2
                        if dim == 1
                            @test_throws BoundsError t[1, 2]
                        elseif dim == 2
                            @test_throws BoundsError t[3, 1]
                        elseif dim == 3
                            @test_throws BoundsError t[3, 4]
                        end
                    else
                        if dim == 1
                            @test_throws BoundsError t[1, 1, 2, 1]
                        elseif dim == 2
                            @test_throws BoundsError t[1, 1, 3, 1]
                        elseif dim == 3
                            @test_throws BoundsError t[1, 4, 3, 1]
                        end
                    end
                end
            end
        end
    end
end
end # of testset


############################
# Trace, norm, det and inv #
############################

@testset "trace, norm, det, inv" begin
for dim in (1,2,3)
    for order in (2,4)
        t = rand(Tensor{order, dim})
        t_sym = rand(Tensor{order, dim})

        if order == 2
            @test (@inferred trace(t)) == sum([t[i,i] for i in 1:dim])
            @test (@inferred trace(t_sym)) == sum([t_sym[i,i] for i in 1:dim])

            @test trace(t) ≈ vol(t) ≈ mean(t)*3.0
            @test trace(t_sym) ≈ vol(t_sym) ≈ mean(t_sym)*3.0

            #@test_approx_eq_eps (mean(dev(t)) / norm(t)) 0.0 1e-14
            #@test_approx_eq_eps (mean(dev(t_sym)) / norm(t_sym)) 0.0 1e-14

            #@inferred mean(dev(t_sym)) / norm(t_sym)
            #@inferred mean(dev(t_sym)) / norm(t_sym)
        elseif order == 4
            @test (@inferred trace(t)) == sum([t[i,i,i,i] for i in 1:dim])
            @test (@inferred trace(t_sym)) == sum([t_sym[i,i,i,i] for i in 1:dim])
        end
   end
end

for dim in (1,2,3)
    for order in (1,2,4)
        t = rand(Tensor{order, dim})

        @test t ≈ extract_components(t)
        @test norm(t) ≈ sqrt(sumabs2(extract_components(t)))

        if order != 1
            t_sym = rand(SymmetricTensor{order, dim})
            @test t_sym ≈ extract_components(t_sym)
            @test norm(t_sym) ≈ sqrt(sumabs2(extract_components(t_sym)))
        end

        if order == 2
            @test det(t) ≈ det(extract_components(t))
            @test det(t_sym) ≈ det(extract_components(t_sym))
            @test inv(t) ≈ inv(extract_components(t))
            @test inv(t_sym) ≈ inv(extract_components(t_sym))
        end

   end
end

end # of testset


##############
# Identities #
##############

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
@testset "Tensor identities" begin
for dim in (1,2,3)
    # Identities with second order and first order
    A = rand(Tensor{2, dim})
    B = rand(Tensor{2, dim})
    C = rand(Tensor{2, dim})
    I = one(Tensor{2, dim})
    a = rand(Tensor{1, dim, Float64})
    b = rand(Tensor{1, dim, Float64})

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

    @test convert(SymmetricTensor{2, dim}, A) ≈ 0.5(A + A')


    A_sym = rand(SymmetricTensor{2, dim})
    B_sym = rand(SymmetricTensor{2, dim})
    C_sym = rand(SymmetricTensor{2, dim})
    I_sym = one(SymmetricTensor{2, dim})

    @test A_sym ⊡ I_sym ≈ trace(A_sym)
    @test det(A_sym) ≈ det(A_sym.')

    @test (I_sym ⊗ I_sym) ⊡ A_sym ≈ trace(A_sym) * I_sym
    @test ((I_sym ⊗ I_sym) ⊡ A_sym) ⊡ A_sym ≈ trace(A_sym)^2
end

for dim in (1,2,3)
    # Identities with second order and first order
    II = one(Tensor{4, dim})
    I = one(Tensor{2, dim})
    A = rand(Tensor{2, dim})
    #II_sym = one(SymmetricTensor{4, dim})
    #A_sym = rand(SymmetricTensor{2, dim})
    #I_sym = one(SymmetricTensor{2, dim})

    @test II ⊡ A ≈ A
    @test A ⊡ II ≈ A
    @test II ⊡ A ⊡ A ≈ (trace(A.' ⋅ A))

    #@test II_sym ⊡ A_sym ≈ A_sym
    #@test A_sym ⊡ II_sym ≈ A_sym
end

end # of testset


########################
# Promotion/Conversion #
########################
@testset "promotion/conversion" begin
const T = Float32
const WIDE_T = widen(T)
for dim in (1,2,3)
    for order in (1,2,4)

        tens = Tensor{order, dim, T, dim^order}
        tens_wide = Tensor{order, dim, WIDE_T, dim^order}

        @test promote_type(tens, tens) == tens
        @test promote_type(tens_wide, tens) == tens_wide
        @test promote_type(tens, tens_wide) == tens_wide

        A = rand(Tensor{order, dim, T})
        B = rand(Tensor{order, dim, WIDE_T})
        @test typeof(A + B) == tens_wide

        if order != 1

            M = ContMechTensors.n_components(SymmetricTensor{order, dim})
            sym = SymmetricTensor{order, dim, T, M}
            sym_wide = SymmetricTensor{order, dim, WIDE_T, M}

            @test promote_type(sym, sym) == sym
            @test promote_type(sym_wide, sym_wide) == sym_wide
            @test promote_type(sym, sym_wide) == sym_wide
            @test promote_type(sym_wide, sym) == sym_wide

            @test promote_type(sym, tens) == tens
            @test promote_type(sym_wide, tens_wide) == tens_wide
            @test promote_type(tens, sym_wide) == tens_wide

            A = rand(SymmetricTensor{order, dim, T})
            B = rand(SymmetricTensor{order, dim, WIDE_T})
            @test typeof(A + B) == sym_wide
        end
    end
end

end  # of testset
