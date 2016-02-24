using ContMechTensors
using Base.Test

import ContMechTensors: n_independent_components, ArgumentError, get_data

const T = Float32


for dim in (1,2,3)
    for order in (1,2,4)
        n_sym = n_independent_components(dim, true)
        n = n_independent_components(dim, false)

        if order == 1
            v = rand(dim)
            v_err = rand(dim + 1)
        elseif order == 2
            v = rand(n)
            v_sym = rand(n_sym)
            v_err = rand(n+1)
            v_sym_err = rand(n_sym+1)
        elseif order == 4
            v = rand(n,n)
            v_sym = rand(n_sym, n_sym)
            v_err = rand(n+1, n+1)
            v_sym_err = rand(n_sym+1, n_sym+1)
        end

        ################
        # Constructors #
        ################


        t = @inferred Tensor{order, dim}(copy(v))
        @test get_data(t) == v
        @test_throws ArgumentError t = Tensor{order, dim}(v_err)

        if order != 1
            t_sym =  @inferred SymmetricTensor{order, dim}(copy(v_sym))
            @test get_data(t_sym) == v_sym
            @test_throws ArgumentError t_sym = SymmetricTensor{order, dim}(v_sym_err)
        end

        #####################################
        # copy, copy!, isequal, ==, similar #
        #####################################

        t_copy = @inferred copy(t)
        @test t_copy == t
        t_sim = @inferred similar(t)
        copy!(t_sim, t)
        @test t_sim == t

        if order != 1
            t_sym_copy =  @inferred copy(t_sym)
            @test t_sym_copy == t_sym
            t_sym_sim =  @inferred similar(t_sym)
            copy!(t_sym_sim, t_sym)
            @test t_sym_sim == t_sym
        end


        ###############
        # Simple math #
        ###############

        t = Tensor{order, dim}(copy(v))
        t_one = one(t)

        @test (@inferred t + t) == 2*t
        @test (@inferred -t) == zero(t) - t
        @test 2*t == t*2
        @test 0.5 * t ≈ t / 2.0
        @test (@inferred rand(t) * 0.0) == zero(t)

        if order != 1
            t_sym =  @inferred SymmetricTensor{order, dim}(copy(v_sym))
            @test (@inferred t_sym + t_sym) == 2*t_sym
            @test (@inferred -t_sym) == zero(t_sym) - t_sym
            @test (@inferred 2*t_sym) == t_sym*2

            @test 2*t_sym == t_sym*2

            @test (@inferred rand(t_sym) * 0.0) == zero(t_sym)
        end
    end
end

############
# Indexing #
############
for dim in (1,2,3)
    for order in (1,2,4)
        n_sym = n_independent_components(dim, true)
        n = n_independent_components(dim, false)

        if order == 1
            vec = Tensor{order, dim}(rand(dim))
            if dim == 1
                @test vec[:x] == vec[1]
                @test_throws BoundsError vec[:y]
                @test_throws BoundsError vec[2]
                vec[:x] = 2.0
                @test vec[1] == 2.0
            elseif dim == 2
                @test vec[:y] == vec[2]
                @test vec[:x] == vec[1]
                @test_throws BoundsError vec[:z]
                @test_throws BoundsError vec[3]
                vec[:y] = 2.0
                @test vec[2] == 2.0
            elseif dim == 3
                @test vec[:x] == vec[1]
                @test vec[:z] == vec[3]
                @test_throws BoundsError vec[:a]
                @test_throws BoundsError vec[4]
                vec[:z] = 2.0
                @test vec[3] == 2.0
            else
                for t in (SymmetricTensor(rand(n), Val{dim}), Tensor(rand(n), Val{dim}))
                    if order == 2
                        if dim == 1
                            @test t[:x, :x] == t[1,1]
                            @test_throws BoundsError t[:x, :y]
                            @test_throws BoundsError t[1, 2]
                            t[:x, :x] = 2.0
                            @test t[1, 1] == 2.0
                        elseif dim == 2
                            @test t[:x, :y] == t[1,2]
                            @test t[:y, :y] == t[2,2]
                            @test_throws BoundsError t[:z, :y]
                            @test_throws BoundsError t[3, 1]
                            t[:y, :y] = 2.0
                            @test t[2,2] == 2.0
                        elseif dim == 3
                            @test t[:x, :z] == t[1, 3]
                            @test t[:z, :y] == t[3, 2]
                            @test_throws BoundsError t[:a, :y]
                            @test_throws BoundsError t[3, 4]
                            t[:z, :y] = 2.0
                            @test t[3,2] == 2.0
                        end
                    else
                        if dim == 1
                            @test t[:x, :x, :x, :x] == t[1,1,1,1]
                            @test_throws BoundsError t[:x, :y, :x, :x]
                            @test_throws BoundsError t[1, 1, 2, 1]
                            t[:x, :x, :x, :x] = 2.0
                            @test t[1,1,1,1] == 2.0
                        elseif dim == 2
                            @test t[:x, :y, :y, :x] == t[1,2,2,1]
                            @test t[:y, :x, :y, :x] == t[2,1,2,1]
                            @test_throws BoundsError t[:x, :y, :x, :z]
                            @test_throws BoundsError t[1, 1, 3, 1]
                            t[:y, :x, :y, :x] = 2.0
                            @test t[2,1,2,1] == 2.0
                        elseif dim == 3
                            @test t[:z, :y, :y, :x] == t[3,2,2,1]
                            @test t[:y, :x, :y, :z] == t[2,1,2,3]
                            @test_throws BoundsError t[:x, :y, :x, :a]
                            @test_throws BoundsError t[1, 4, 3, 1]
                            t[:y, :x, :y, :z] = 2.0
                            @test t[2,1,2,3] == 2.0
                        end
                    end
                end
            end
        end
    end
end


for dim in (1,2,3)
    for order in (2,4)
        t = rand(Tensor{order, dim})
        t_sym = rand(Tensor{order, dim})

        if order == 2
            @test (@inferred trace(t)) == sum([t[i,i] for i in 1:dim])
            @test (@inferred trace(t_sym)) == sum([t_sym[i,i] for i in 1:dim])

            @test_approx_eq_eps (mean(dev(t)) / norm(t)) 0.0 1e-14
            @test_approx_eq_eps (mean(dev(t_sym)) / norm(t_sym)) 0.0 1e-14

            @inferred mean(dev(t_sym)) / norm(t_sym)
            @inferred mean(dev(t_sym)) / norm(t_sym)
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


##############
# Identities #
##############

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
for dim in (1,2,3)
     println(dim)
    # Identities with second order and first order
    A = rand(Tensor{2, dim})
    B = rand(Tensor{2, dim})
    C = rand(Tensor{2, dim})
    I = one(Tensor{2, dim})
    a = rand(Tensor{1, dim, Float64})
    b = rand(Tensor{1, dim, Float64})

    @test A * B ≈ (A' ⋅ B) * one(A)
    @test A * (a ⊗ b) ≈ (A ⋅ b) ⋅ a
    @test (A ⋅ a) ⋅ (B ⋅ b) ≈ (A.' ⋅ B) * (a ⊗ b)
    @test (A ⋅ a) ⊗ b ≈ A ⋅ (a ⊗ b)
    @test a ⊗ (A ⋅ b) ≈ (A ⋅ (b ⊗ a)).'
    @test a ⊗ (A ⋅ b) ≈ (a ⊗ b) ⋅ A'

    @test A * I ≈ trace(A)
    @test det(A) ≈ det(A.')
    @test trace(inv(A) ⋅ A) ≈ dim
    @test inv(A) ⋅ A ≈ I

    @test (I ⊗ I) * A ≈ trace(A) * I
    @test (I ⊗ I) * A * A ≈ trace(A)^2


    A_sym = rand(SymmetricTensor{2, dim})
    B_sym = rand(SymmetricTensor{2, dim})
    C_sym = rand(SymmetricTensor{2, dim})
    I_sym = one(SymmetricTensor{2, dim})

    @test A_sym * I_sym ≈ trace(A_sym)
    @test det(A_sym) ≈ det(A_sym.')

    @test (I_sym ⊗ I_sym) * A_sym ≈ trace(A_sym) * I_sym
    @test (I_sym ⊗ I_sym) * A_sym * A_sym ≈ trace(A_sym)^2
end

for dim in (1,2,3)
    println(dim)
    # Identities with second order and first order
    II = one(Tensor{4, dim})
    I = one(Tensor{2, dim})
    A = rand(Tensor{2, dim})
    II_sym = one(SymmetricTensor{4, dim})
    A_sym = rand(SymmetricTensor{2, dim})
    I_sym = one(SymmetricTensor{2, dim})

    @test II * A ≈ A
    @test A * II ≈ A
    @test II * A * A ≈ (trace(A.' ⋅ A))

    @test II_sym * A_sym ≈ A_sym
    @test A_sym * II_sym ≈ A_sym

end

########################
# Promotion/Conversion #
########################

const WIDE_T = widen(T)
for dim in (1,2,3)
    for order in (1,2,4)
        if order == 1
            M = 1
        else
            M = div(order, 2)
        end
        tens = Tensor{order, dim, T, M}
        tens_wide = Tensor{order, dim, WIDE_T, M}

        @test promote_type(tens, tens) == tens
        @test promote_type(tens_wide, tens) == tens_wide
        @test promote_type(tens, tens_wide) == tens_wide

        A = rand(Tensor{order, dim, T})
        B = rand(Tensor{order, dim, WIDE_T})
        @test typeof(A + B) == tens_wide

        if order != 1
            sym = SymmetricTensor{order, dim, T, M}
            sym_wide = SymmetricTensor{order, dim, WIDE_T, M}

            @test promote_type(sym, sym) == sym
            @test promote_type(sym_wide, sym_wide) == sym_wide
            @test promote_type(sym, sym_wide) == sym_wide
            @test promote_type(sym_wide, sym) == sym_wide

            A = rand(SymmetricTensor{order, dim, T})
            B = rand(SymmetricTensor{order, dim, WIDE_T})
            @test typeof(A + B) == sym_wide
        end
    end
end
