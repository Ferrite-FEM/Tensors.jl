using ContMechTensors
using Base.Test

import ContMechTensors: n_independent_components, InternalError, get_data

const T = Float32

########################
# Promotion/Conversion #
########################

const WIDE_T = widen(T)
for dim in (1,2,3)
    for order in (2,4)
        tens = Tensor{order, dim, T, div(order,2)}
        tens_wide = Tensor{order, dim, WIDE_T, div(order,2)}

        @test promote_type(tens, tens) == tens
        @test promote_type(tens_wide, tens) == tens_wide
        @test promote_type(tens, tens_wide) == tens_wide

        sym = SymmetricTensor{order, dim, T, div(order,2)}
        sym_wide = SymmetricTensor{order, dim, WIDE_T, div(order,2)}

        @test promote_type(sym, sym) == sym
        @test promote_type(sym_wide, sym_wide) == sym_wide
        @test promote_type(sym, sym_wide) == sym_wide
        @test promote_type(sym_wide, sym) == sym_wide

        # No automatic promotion between symmetric and nonsymmetric
        #@test promote_type(tens, sym) == tens
        #@test promote_type(tens_wide, sym) == tens_wide
        #@test promote_type(tens, sym_wide) == tens_wide
        #@test promote_type(sym_wide, tens) == tens_wide
        #@test promote_type(tens_wide, sym_wide) == tens_wide
    end
end



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

        if order == 1
            t = @inferred Vec(copy(v), Val{dim})
            @test get_data(t) == v
            @test_throws InternalError t = Vec(v_err, Val{dim})
        else
          t = @inferred Tensor(copy(v), Val{dim})
          @test get_data(t) == v

          t_sym =  @inferred SymmetricTensor(copy(v_sym), Val{dim})
          @test get_data(t_sym) == v_sym
           @test_throws InternalError t = Tensor(v_err, Val{dim})
           @test_throws InternalError t_sym = SymmetricTensor(v_sym_err, Val{dim})
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

        if order == 1
            t =  Vec(copy(v), Val{dim})
            t_one =  one(Vec{dim, Float64})
        else
            t =  @inferred Tensor(copy(v), Val{dim})
            t_one =  @inferred one(Tensor{order, dim, Float64})
            t_one_sym =  @inferred one(SymmetricTensor{order, dim, Float64})
        end

        @test (@inferred t + t) == 2*t
        @test (@inferred -t) == zero(t) - t
        @test 2*t == t*2
        @test 0.5 * t ≈ t / 2.0
        @test (@inferred rand(t) * 0.0) == zero(t)

        if order != 1
            t_sym =  @inferred SymmetricTensor(copy(v_sym), Val{dim})
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
            vec = Vec(rand(dim), Val{dim})
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
        #dcontract(S1, S2) ≈ dcontract(dot(transpose(S1), S2), one(S1))
   end
end

for dim in (1,2,3)
    for order in (1,2,4)
        if order == 1
            t = rand(Vec{dim})
        else
            t = rand(Tensor{order, dim})
            t_sym = rand(SymmetricTensor{order, dim})
        end

        @test t ≈ extract_components(t)
        @test norm(t) ≈ sqrt(sumabs2(extract_components(t)))

        if order != 1
            @test t_sym ≈ extract_components(t_sym)
            @test norm(t_sym) ≈ sqrt(sumabs2(extract_components(t_sym)))
        end
   end
end


##############
# Identities #
##############

# https://en.wikiversity.org/wiki/Continuum_mechanics/Tensor_algebra_identities
for dim in (1,2,3)
    A = rand(Tensor{2, dim})
    B = rand(Tensor{2, dim})
    a = rand(Vec{dim, Float64})
    b = rand(Vec{dim, Float64})

    @test A * B ≈ (A' ⋅ B) * one(A)
    @test A * (a ⊠ b) ≈ (A ⋅ b) ⋅ a
    @test (A ⋅ a) ⋅ (B ⋅ b) ≈ (A.' ⋅ B) * (a ⊠ b)
    @test (A ⋅ a) ⊗ b ≈ A ⋅ (a ⊗ b)
    @test a ⊗ (A ⋅ b) ≈ (A ⋅ (b ⊗ a)).'
    @test a ⊗ (A ⋅ b) ≈ (a ⊗ b) ⋅ A'
end


#
#    dim = 2
#     Ee = rand(SymmetricTensor{4, dim})
#
#     E = 200e9
#     ν = 0.3
#     μ = E / (2(1+ν))
#     λ = ν*E / ((1+ν) * (1-2ν))
#
#     for i in 1:dim, j in 1:dim, k in 1:dim, l in 1:dim
#        v = ((i == k && j == l ? μ : 0.0) +
#            (i == l && j == k ? μ : 0.0) +
#            (i == j && k == l ? λ : 0.0))
#         Ee[i,j,k,l] = v
#      end
#
#
#
#      ε * (Ee * ε) ==
#
#
#tmp[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
#                             ((i==l) && (j==k) ? mu : 0.0) +
#                             ((i==j) && (k==l) ? lambda : 0.0));
#
#for dim in (1,2,3)
#    for T in (Float32, Float64)
#        println(dim)
#        println(T)
#        n = n_independent_components(SymmetricTensor{2, dim})
#        t = SymmetricTensor(rand(n), Val{dim})
#        S2 = SymmetricTensor(rand(n), Val{dim})
#        S3 = copy(S2)
#        @test S3 == S2
#        copy!(S1, S3)
#        @test S1 == S3
#        S4 = similar(S1)
#        @test size(S4) == size(S1)
#
#        if dim == 1
#            @test S1[:x, :x] == S1[1,1]
#        elseif dim == 2
#            @test S1[:x, :y] == S1[1,2]
#            @test S1[:y, :y] == S1[2,2]
#        elseif dim == 3
#            @test S3[:x, :z] == S3[1, 3]
#            @test S3[:z, :y] == S3[3, 2]
#        end
#
#        @test issym(S1) == true
#        @test transpose(S1) == S1
#
#        @test size(S1) == (dim, dim)
#
#        @test one(S1) == one(SymmetricTensor{2, dim, T})
#        @test zero(S1) == zero(SymmetricTensor{2, dim, T})
#
#        @test dcontract(S1, one(S1)) == trace(S1)
#        #@test dot(S1, one(S1)) == S1
#        v = rand(dim)
#        @test dot(one(S1), v) == v
#        # A : B = (A' : B) : 1
#        #@test dcontract(S1, S2) ≈ dcontract(dot(transpose(S1), S2), one(S1))
#        # A : (a ⊗ b) = (A ⋅ b) ⋅ a
#
#        S1 = SymmetricTensor(rand(n), Val{dim})
#        S2 = SymmetricTensor(rand(n), Val{dim})
#        S3 = SymmetricTensor(rand(n), Val{dim})
#
#        S = S1 + S2
#        S = S1 - S2
#        S = 2*S1
#        S = -S1
#        S = S1*2
#      #  @test dcontract(oprod(S1, S2), S3) ≈ S1 * dcontract(W, V)
#
#    end
#end
#
#T = Float64
#dim = 2
#S1 = SymmetricTensor(rand(3), Val{dim})
#S2 = SymmetricTensor(rand(3), Val{dim})

#include("test_transformations.jl")
#include("test_ops.jl")


