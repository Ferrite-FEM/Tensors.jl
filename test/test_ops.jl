# Test dcontract
using ContMechTensors
using Base.Test

# Dcontract
for dim in (1,2,3)
    println(dim)
    AA = rand(Tensor{4, dim})
    BB = rand(Tensor{4, dim})
    A = rand(Tensor{2, dim})
    B = rand(Tensor{2, dim})
    a = rand(Tensor{1, dim})
    b = rand(Tensor{1, dim})

    AA_sym = rand(SymmetricTensor{4, dim})
    BB_sym = rand(SymmetricTensor{4, dim})
    A_sym = rand(SymmetricTensor{2, dim})
    B_sym = rand(SymmetricTensor{2, dim})


    #############
    # Dcontract #
    #############

    # 4 - 4
     @test vec(dcontract(AA, BB)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
     @test vec(dcontract(AA_sym, BB_sym)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))
     @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym)) ≈ dcontract(AA_sym, BB_sym)
     @test vec(dcontract(AA_sym, BB)) ≈ vec(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(BB), (dim^2, dim^2)))
     @test vec(dcontract(AA, BB_sym)) ≈ vec(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(BB_sym), (dim^2, dim^2)))

     # 2 - 4
     @test dcontract(AA, A) ≈ reshape(reshape(vec(AA), (dim^2, dim^2)) * reshape(vec(A), (dim^2,)), dim, dim)
     @test dcontract(A, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
     @test dcontract(AA_sym, A_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2)) * reshape(vec(A_sym), (dim^2,)), dim, dim)
     @test dcontract(A_sym, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
     @test dcontract(A, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
     @test dcontract(A_sym, AA) ≈ reshape(reshape(vec(AA), (dim^2, dim^2))' * reshape(vec(A_sym), (dim^2,)), dim, dim)
     @test dcontract(A, AA_sym) ≈ reshape(reshape(vec(AA_sym), (dim^2, dim^2))' * reshape(vec(A), (dim^2,)), dim, dim)
     @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym)) ≈ dcontract(AA_sym, A_sym)


     # 2 - 2
     @test dcontract(A, B) ≈ sum(vec(A) .* vec(B))
     @test dcontract(A_sym, B_sym) ≈ sum(vec(A_sym) .* vec(B_sym))
     @test dcontract(A, B_sym) ≈ sum(vec(A) .* vec(B_sym))
     @test dcontract(A_sym, B) ≈ sum(vec(A_sym) .* vec(B))


     #################
     # Outer product #
     #################

     @test otimes(a, b) ≈ extract_components(a) * extract_components(b)'
     @test reshape(vec(otimes(A, B)), dim^2, dim^2) ≈ vec(A) * vec(B)'
     @test reshape(vec(otimes(A_sym, B_sym)), dim^2, dim^2) ≈ vec(A_sym) * vec(B_sym)'
     @test reshape(vec(otimes(A, B_sym)), dim^2, dim^2) ≈ vec(A) * vec(B_sym)'
     @test reshape(vec(otimes(A_sym, B)), dim^2, dim^2) ≈ vec(A_sym) * vec(B)'

    ################
    # Dot products #
    ################

    # 1 - 2
    @test dot(a, b) ≈ sum(extract_components(a) .* extract_components(b))
    @test dot(A, b) ≈ reshape(vec(A), (dim,dim)) * extract_components(b)
    @test dot(b, A) ≈ reshape(vec(A), (dim,dim))' * extract_components(b)
    @test dot(A_sym, b) ≈ reshape(vec(A_sym), (dim,dim)) * extract_components(b)
    @test dot(b, A_sym) ≈ reshape(vec(A_sym), (dim,dim))' * extract_components(b)

    # 2 - 2
    @test dot(A, B) ≈ reshape(vec(A), (dim,dim))  * reshape(vec(B), (dim,dim))
    @test dot(A_sym, B_sym) ≈ reshape(vec(A_sym), (dim,dim))  * reshape(vec(B_sym), (dim,dim))
    @test dot(A, B_sym) ≈ reshape(vec(A), (dim,dim))  * reshape(vec(B_sym), (dim,dim))
    @test dot(A_sym, B) ≈ reshape(vec(A_sym), (dim,dim))  * reshape(vec(B), (dim,dim))
    @test tdot(A) ≈ reshape(vec(A), (dim,dim))'  * reshape(vec(A), (dim,dim))

    ###############
    # Determinant #
    ###############

    @test det(A) ≈ det(reshape(vec(A), (dim,dim)))
    @test det(A_sym) ≈ det(reshape(vec(A_sym), (dim,dim)))

    ##########################
    # Creating with function #
    ##########################

    fi = (i) -> cos(i)
    fij = (i,j) -> cos(i) + sin(j)
    fijkl = (i, j, k ,l) -> cos(i) + sin(j) + tan(k) + exp(l)

    af = Tensor{1,dim}(fi)
    Af = Tensor{2,dim}(fij)
    AAf = Tensor{4,dim}(fijkl)

    Af_sym = SymmetricTensor{2,dim}(fij)
    AAf_sym = SymmetricTensor{4,dim}(fijkl)
    for i in 1:dim
        @test af[i] == fi(i)
        for j in 1:dim
            @test Af[i,j] == fij(i, j)
            for k in 1:dim, l in 1:dim
                @test AAf[i,j,k,l] == fijkl(i, j,k,l)

            end
        end
    end

    for i in 1:dim, j in 1:i
        @test Af_sym[i,j] == fij(i, j)
        for k in 1:dim, l in 1:k
             @test AAf_sym[i,j,k,l] == fijkl(i, j,k,l)
        end
    end



end


