function _permutedims(S::FourthOrderTensor{dim}, idx::NTuple{4,Int}) where dim
    sort([idx...]) == [1,2,3,4] || throw(ArgumentError("Missing index."))
    neworder = sortperm([idx...])
    f = (i,j,k,l) -> S[[i,j,k,l][neworder]...]
    return Tensor{4,dim}(f)
end

@testsection "tensor ops" begin
for T in (Float32, Float64, F64), dim in (1,2,3)
println("T = $T, dim = $dim")
AA = rand(Tensor{4, dim, T})
BB = rand(Tensor{4, dim, T})
A = rand(Tensor{2, dim, T})
B = rand(Tensor{2, dim, T})
a = rand(Tensor{1, dim, T})
b = rand(Tensor{1, dim, T})

AA_sym = rand(SymmetricTensor{4, dim, T})
BB_sym = rand(SymmetricTensor{4, dim, T})
A_sym = rand(SymmetricTensor{2, dim, T})
B_sym = rand(SymmetricTensor{2, dim, T})
symA = symmetric(A)
symB = symmetric(B)

i,j,k,l = rand(1:dim,4)

@testsection "double contraction" begin
    # 4 - 4
    @test vec((@inferred dcontract(AA, BB))::Tensor{4, dim, T})                  ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA, BB_sym))::Tensor{4, dim, T})              ≈ vec(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test vec((@inferred dcontract(AA_sym, BB_sym))::SymmetricTensor{4, dim, T}) ≈ vec(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(BB_sym), (dim^2, dim^2))))
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, BB_sym))            ≈ dcontract(AA_sym, BB_sym)

    # 2 - 4
    @test (@inferred dcontract(AA, A))::Tensor{2, dim, T}                  ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA_sym, A))::SymmetricTensor{2, dim, T}     ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA, A_sym))::Tensor{2, dim, T}              ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(AA_sym, A_sym))::SymmetricTensor{2, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))) * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A, AA))::Tensor{2, dim, T}                  ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A_sym, AA))::Tensor{2, dim, T}              ≈ reshape(collect(reshape(vec(AA), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A, AA_sym))::SymmetricTensor{2, dim, T}     ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A), (dim^2,))), dim, dim)
    @test (@inferred dcontract(A_sym, AA_sym))::SymmetricTensor{2, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^2, dim^2))') * collect(reshape(vec(A_sym), (dim^2,))), dim, dim)
    @test dcontract(convert(Tensor, AA_sym), convert(Tensor, A_sym))       ≈ dcontract(AA_sym, A_sym)

    # 2 - 2
    @test (@inferred dcontract(A, B))::T         ≈ sum(vec(A) .* vec(B))
    @test (@inferred dcontract(A_sym, B))::T     ≈ sum(vec(A_sym) .* vec(B))
    @test (@inferred dcontract(A, B_sym))::T     ≈ sum(vec(A) .* vec(B_sym))
    @test (@inferred dcontract(A_sym, B_sym))::T ≈ sum(vec(A_sym) .* vec(B_sym))
end # of testsection

@testsection "outer products (otimes, otimesu, otimesl)" begin
    # binary otimes
    @test             (@inferred otimes(a, b))::Tensor{2, dim, T}                                  ≈ Array(a) * Array(b)'
    @test reshape(vec((@inferred otimes(A, B))::Tensor{4, dim, T}), dim^2, dim^2)                  ≈ vec(A) * vec(B)'
    @test reshape(vec((@inferred otimes(A_sym, B))::Tensor{4, dim, T}), dim^2, dim^2)              ≈ vec(A_sym) * vec(B)'
    @test reshape(vec((@inferred otimes(A, B_sym))::Tensor{4, dim, T}), dim^2, dim^2)              ≈ vec(A) * vec(B_sym)'
    @test reshape(vec((@inferred otimes(A_sym, B_sym))::SymmetricTensor{4, dim, T}), dim^2, dim^2) ≈ vec(A_sym) * vec(B_sym)'
    # unary otimes
    @test (@inferred otimes(a))::SymmetricTensor{2, dim, T} ≈ otimes(a, a)

    # binary otimesu
    @test (@inferred otimesu(A, B))::Tensor{4, dim, T}         ≈ _permutedims(otimes(A, B), (1,3,2,4))
    @test (@inferred otimesu(A_sym, B))::Tensor{4, dim, T}     ≈ _permutedims(otimes(A_sym, B), (1,3,2,4))
    @test (@inferred otimesu(A, B_sym))::Tensor{4, dim, T}     ≈ _permutedims(otimes(A, B_sym), (1,3,2,4))
    @test (@inferred otimesu(A_sym, B_sym))::Tensor{4, dim, T} ≈ _permutedims(otimes(A_sym, B_sym), (1,3,2,4))

    # binary otimesl
    @test (@inferred otimesl(A, B))::Tensor{4, dim, T}         ≈ _permutedims(otimes(A, B), (1,3,4,2))
    @test (@inferred otimesl(A_sym, B))::Tensor{4, dim, T}     ≈ _permutedims(otimes(A_sym, B), (1,3,4,2))
    @test (@inferred otimesl(A, B_sym))::Tensor{4, dim, T}     ≈ _permutedims(otimes(A, B_sym), (1,3,4,2))
    @test (@inferred otimesl(A_sym, B_sym))::Tensor{4, dim, T} ≈ _permutedims(otimes(A_sym, B_sym), (1,3,4,2))
end # of testsection

@testsection "dot products" begin
    # 1 - 1
    @test (@inferred dot(a, b))::T               ≈ sum(Array(a) .* Array(b))
    # 1 - 2
    @test (@inferred dot(A, b))::Vec{dim, T}     ≈ Array(A) * Array(b)
    @test (@inferred dot(A_sym, b))::Vec{dim, T} ≈ Array(A_sym) * Array(b)
    @test (@inferred dot(a, B))::Vec{dim, T}     ≈ Array(B)' * Array(a)
    @test (@inferred dot(a, B_sym))::Vec{dim, T} ≈ Array(B_sym)' * Array(a)
    # 2 - 2
    # binary
    @test (@inferred dot(A, B))::Tensor{2, dim, T}         ≈ Array(A) * Array(B)
    @test (@inferred dot(A_sym, B))::Tensor{2, dim, T}     ≈ Array(A_sym) * Array(B)
    @test (@inferred dot(A, B_sym))::Tensor{2, dim, T}     ≈ Array(A) * Array(B_sym)
    @test (@inferred dot(A_sym, B_sym))::Tensor{2, dim, T} ≈ Array(A_sym) * Array(B_sym)
    # unary
    @test (@inferred dot(A_sym))::SymmetricTensor{2, dim, T}  ≈ dot(A_sym, A_sym)
    @test (@inferred tdot(A))::SymmetricTensor{2, dim, T}     ≈ dot(transpose(A), A)
    @test (@inferred tdot(A_sym))::SymmetricTensor{2, dim, T} ≈ dot(transpose(A_sym), A_sym)
    @test (@inferred dott(A))::SymmetricTensor{2, dim, T}     ≈ dot(A, transpose(A))
    @test (@inferred dott(A_sym))::SymmetricTensor{2, dim, T} ≈ dot(A_sym, transpose(A_sym))
    # 2 - 4
    @test (@inferred dot(AA, B))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA), (dim^3, dim))) * collect(reshape(vec(B), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B, AA))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B), (dim, dim))) * collect(reshape(vec(AA), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA_sym, B))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^3, dim))) * collect(reshape(vec(B), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B, AA_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B), (dim, dim))) * collect(reshape(vec(AA_sym), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA, B_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA), (dim^3, dim))) * collect(reshape(vec(B_sym), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B_sym, AA))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B_sym), (dim, dim))) * collect(reshape(vec(AA), (dim, dim^3))), (dim, dim, dim, dim))
    @test (@inferred dot(AA_sym, B_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(AA_sym), (dim^3, dim))) * collect(reshape(vec(B_sym), (dim, dim))), (dim, dim, dim, dim))
    @test (@inferred dot(B_sym, AA_sym))::Tensor{4, dim, T} ≈ reshape(collect(reshape(vec(B_sym), (dim, dim))) * collect(reshape(vec(AA_sym), (dim, dim^3))), (dim, dim, dim, dim))
end # of testsection

@testsection "symmetric/skew-symmetric" begin
    if dim == 1 # non-symmetric tensors are symmetric
        @test (@inferred issymmetric(A))
        @test (@inferred issymmetric(AA))
    elseif dim != 1
        @test !(@inferred issymmetric(A))
        @test !(@inferred issymmetric(AA))
        @test !(@inferred ismajorsymmetric(AA))
        @test !(@inferred isminorsymmetric(AA))
        @test_throws InexactError convert(typeof(A_sym),A)
        @test_throws InexactError convert(typeof(AA_sym),AA)
    end
    @test (@inferred issymmetric(A_sym))
    @test (@inferred issymmetric(AA_sym))
    @test (@inferred isminorsymmetric(AA_sym))
    @test (@inferred issymmetric(symmetric(A)))
    @test (@inferred issymmetric(A + A'))

    @test (@inferred symmetric(A))::SymmetricTensor{2, dim, T}     ≈ 0.5(A + A')
    @test (@inferred symmetric(A_sym))::SymmetricTensor{2, dim, T} ≈ A_sym
    @test convert(typeof(A_sym),convert(Tensor,symmetric(A))) ≈ symmetric(A)

    @test (@inferred symmetric(AA))::SymmetricTensor{4, dim, T} ≈ (@inferred minorsymmetric(AA))::SymmetricTensor{4, dim, T}
    @test minorsymmetric(AA)[i,j,k,l] ≈ minorsymmetric(AA)[j,i,l,k]
    @test issymmetric(convert(Tensor,minorsymmetric(AA)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA)))
    @test (@inferred symmetric(AA_sym))::SymmetricTensor{4, dim, T} ≈ (@inferred minorsymmetric(AA_sym))::SymmetricTensor{4, dim, T}
    @test minorsymmetric(AA_sym)[i,j,k,l] ≈ minorsymmetric(AA_sym)[j,i,l,k]
    @test minorsymmetric(AA_sym) ≈ AA_sym
    @test issymmetric(convert(Tensor,minorsymmetric(AA_sym)))
    @test isminorsymmetric(convert(Tensor,minorsymmetric(AA_sym)))
    @test minorsymmetric(A⊗B) ≈ symA⊗symB

    @test convert(typeof(AA_sym),convert(Tensor,minorsymmetric(AA))) ≈ minorsymmetric(AA)

    @test ((@inferred majorsymmetric(AA))::Tensor{4, dim, T})[i,j,k,l] ≈ 0.5*(AA[i,j,k,l] + AA[k,l,i,j])
    @test majorsymmetric(AA)[i,j,k,l] ≈ majorsymmetric(AA)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA))
    @test ((@inferred majorsymmetric(AA_sym))::SymmetricTensor{4, dim, T})[i,j,k,l] ≈ 0.5*(AA_sym[i,j,k,l] + AA_sym[k,l,i,j])
    @test majorsymmetric(AA_sym)[i,j,k,l] ≈ majorsymmetric(AA_sym)[k,l,i,j]
    @test ismajorsymmetric(majorsymmetric(AA_sym))

    @test (@inferred skew(A))::Tensor{2, dim, T} ≈ 0.5(A - A')
    @test (@inferred skew(A_sym))::Tensor{2, dim, T} ≈ zero(A_sym)

    # Identities
    @test A ≈ symmetric(A) + skew(A)
    @test skew(A) ≈ -skew(A)'
    @test tr(skew(A)) ≈ 0.0
    @test tr(symmetric(A)) ≈ tr(A)
end # of testsection

@testsection "transpose" begin
    @test (@inferred transpose(A))::Tensor{2, dim, T} ≈ Array(A)'
    @test transpose(transpose(A)) ≈ A
    @test (@inferred transpose(A_sym))::SymmetricTensor{2, dim, T} ≈ A_sym ≈ Array(A_sym)'
    @test transpose(transpose(A_sym)) ≈ A_sym

    @test (@inferred transpose(AA))::Tensor{4, dim, T} ≈ (@inferred minortranspose(AA))::Tensor{4, dim, T}
    @test AA[i,j,k,l] ≈ minortranspose(AA)[j,i,l,k]
    @test AA_sym[i,j,k,l] ≈ minortranspose(AA_sym)[j,i,l,k]
    @test (@inferred transpose(AA_sym))::SymmetricTensor{4, dim, T} ≈ (@inferred minortranspose(AA_sym))::SymmetricTensor{4, dim, T} ≈ AA_sym
    @test minortranspose(minortranspose(AA)) ≈ AA
    @test minortranspose(minortranspose(AA_sym)) ≈ AA_sym

    @test majortranspose((@inferred majortranspose(AA))::Tensor{4, dim, T}) ≈ AA
    @test majortranspose((@inferred majortranspose(AA_sym))::Tensor{4, dim, T}) ≈ AA_sym
    @test AA[i,j,k,l] ≈ majortranspose(AA)[k,l,i,j]
    @test AA_sym[i,j,k,l] ≈ majortranspose(AA_sym)[k,l,i,j]

    @test minortranspose(AA) ≈ _permutedims(AA,(2,1,4,3))
    @test minortranspose(AA_sym) ≈ _permutedims(AA_sym,(2,1,4,3))
    @test majortranspose(AA) ≈ _permutedims(AA,(3,4,1,2))
    @test majortranspose(AA_sym) ≈ _permutedims(AA_sym,(3,4,1,2))
end # of testsection

@testsection "cross product" begin
    @test (@inferred a × a)::Vec{3, T} ≈ Vec{3, T}((0.0,0.0,0.0))
    @test a × b ≈ -b × a
    if dim == 2
        ad = Vec{2, T}((1.0,0.0))
        ad2 = Vec{2, T}((0.0,1.0))
        @test (@inferred ad × ad2)::Vec{3, T} ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
    if dim == 3
        ad = Vec{3, T}((1.0,0.0,0.0))
        ad2 = Vec{3, T}((0.0,1.0,0.0))
        @test (@inferred ad × ad2)::Vec{3, T} ≈ Vec{3, T}((0.0, 0.0, 1.0))
    end
    if T == Float64 # mixed eltype
        @test rand(Vec{dim,Float64}) × rand(Vec{dim,Float32}) isa Vec{3,Float64}
    end
end # of testsection

@testsection "special" begin
    AAT = Tensor{4, dim, T}((i,j,k,l) -> AA_sym[i,l,k,j])
    @test AAT ⊡ (b ⊗ a) ≈ (@inferred dotdot(a, AA_sym, b))::Tensor{2, dim, T}
end # of testsection

if dim == 3
    @testsection "rotation" begin
        x = eᵢ(Vec{3, T}, 1)
        y = eᵢ(Vec{3, T}, 2)
        z = eᵢ(Vec{3, T}, 3)

        @test (@inferred rotate(z, z, rand(T)))::Vec{3, T} ≈ z
        @test rotate(2*z, y, π/2) ≈ 2*x
        @test rotate(3*z, y, π) ≈ -3*z
        @test rotate(x+y+z, z, π/4) ≈ Vec{3}((0.0,√2,1.0))

        @test rotate(a, b, 0) ≈ a
        @test rotate(a, b, π) ≈ rotate(a, b, -π)
        @test rotate(a, b, π/2) ≈ rotate(a, -b, -π/2)

        @test rotate(A, x, π) ≈ rotate(A, x, -π)
        @test rotate(rotate(rotate(A, x, π), y, π), z, π) ≈ A
        @test rotate(A, a, 0) ≈ A
        @test rotate(A, a, π/2) ≈ rotate(A, -a, -π/2)
        A_sym_T = convert(Tensor{2,3}, A_sym)
        @test rotate(A_sym, x, π)::SymmetricTensor ≈ rotate(A_sym, x, -π)::SymmetricTensor ≈ rotate(A_sym_T, x, -π)
        @test rotate(rotate(rotate(A_sym, x, π), y, π), z, π)::SymmetricTensor ≈ A_sym
        @test rotate(A_sym, a, 0) ≈ A_sym
        @test rotate(A_sym, a, π/2) ≈ rotate(A_sym, -a, -π/2)

        @test rotate(AA, x, π)::Tensor ≈ rotate(AA, x, -π)
        @test rotate(rotate(rotate(AA, x, π), y, π), z, π) ≈ AA
        @test rotate(AA, a, 0) ≈ AA
        @test rotate(AA, a, π/2) ≈ rotate(AA, -a, -π/2)
        AA_sym_T = convert(Tensor{4,3}, AA_sym)
        @test rotate(AA_sym, x, π)::SymmetricTensor ≈ rotate(AA_sym, x, -π) ≈ rotate(AA_sym_T, x, π)
        @test rotate(rotate(rotate(AA_sym, x, π), y, π), z, π) ≈ AA_sym
        @test rotate(AA_sym, a, 0) ≈ AA_sym
        @test rotate(AA_sym, a, π/2) ≈ rotate(AA_sym, -a, -π/2)

        v1, v2, v3, v4, axis = [rand(Vec{3,T}) for _ in 1:5]
        α = rand(T) * π
        R = Tensors.rotation_matrix(axis, α)
        v1v2v3v4 = (v1 ⊗ v2) ⊗ (v3 ⊗ v4)
        Rv1v2v3v4 = ((R ⋅ v1) ⊗ (R ⋅ v2)) ⊗ ((R ⋅ v3) ⊗ (R ⋅ v4))
        @test rotate(v1v2v3v4, axis, α) ≈ Rv1v2v3v4
        v1v1v2v2 = otimes(v1) ⊗ otimes(v2)
        Rv1v1v2v2 = otimes(R ⋅ v1) ⊗ otimes(R ⋅ v2)
        @test rotate(v1v1v2v2, axis, α) ≈ Rv1v1v2v2
    end
end

@testsection "tovoigt/fromvoigt" begin
    # https://classes.engineering.wustl.edu/2009/spring/mase5513/abaqus/docs/v6.6/books/usb/default.htm?startat=pt01ch01s02aus02.html
    abaqus = dim == 1 ? fill(1, 1, 1) : dim == 2 ? [1 3; 0 2] : [1 4 6; 0 2 5; 0 0 3]
    T2 = T(2)
    @test (@inferred tovoigt(AA)) * (@inferred tovoigt(A)) ≈ tovoigt(AA ⊡ A)
    @test (@inferred tovoigt(AA_sym)) * (@inferred tovoigt(A_sym, offdiagscale=T2)) ≈ tovoigt(AA_sym ⊡ A_sym)

    nc(x::SecondOrderTensor) = length(x.data)
    nc(x::FourthOrderTensor) = Int(sqrt(length(x.data)))
    x = zeros(T, nc(A) + 1); tovoigt!(x, A; offset=1)
    @test x[2:end] == tovoigt(A)
    x = zeros(T, nc(A_sym) + 1); tovoigt!(x, A_sym; offset=1)
    @test x[2:end] == tovoigt(A_sym)
    x = zeros(T, nc(AA) + 1, nc(AA) + 1); tovoigt!(x, AA; offset_i=1, offset_j=1)
    @test x[2:end, 2:end] == tovoigt(AA)
    x = zeros(T, nc(AA_sym) + 1, nc(AA_sym) + 1); tovoigt!(x, AA_sym; offset_i=1, offset_j=1)
    @test x[2:end, 2:end] == tovoigt(AA_sym)

    x = zeros(T, nc(A) + 1); tomandel!(x, A; offset=1)
    @test x[2:end] == tomandel(A)
    x = zeros(T, nc(A_sym) + 1); tomandel!(x, A_sym; offset=1)
    @test x[2:end] == tomandel(A_sym)
    x = zeros(T, nc(AA) + 1, nc(AA) + 1); tomandel!(x, AA; offset_i=1, offset_j=1)
    @test x[2:end, 2:end] == tomandel(AA)
    x = zeros(T, nc(AA_sym) + 1, nc(AA_sym) + 1); tomandel!(x, AA_sym; offset_i=1, offset_j=1)
    @test x[2:end, 2:end] == tomandel(AA_sym)

    @test (@inferred fromvoigt(Tensor{2,dim}, tovoigt(A))) ≈ A
    @test (@inferred fromvoigt(Tensor{4,dim}, tovoigt(AA))) ≈ AA
    @test (@inferred fromvoigt(SymmetricTensor{2,dim}, tovoigt(A_sym, offdiagscale=T2), offdiagscale=T2)) ≈ A_sym
    @test (@inferred fromvoigt(SymmetricTensor{4,dim}, tovoigt(AA_sym, offdiagscale=T2), offdiagscale=T2)) ≈ AA_sym

    @test (@inferred fromvoigt(SymmetricTensor{2,dim}, tovoigt(A_sym; order=abaqus); order=abaqus)) ≈ A_sym
    @test (@inferred fromvoigt(SymmetricTensor{4,dim}, tovoigt(AA_sym; order=abaqus); order=abaqus)) ≈ AA_sym

    @test (@inferred tomandel(AA_sym)) * (@inferred tomandel(A_sym)) ≈ tomandel(AA_sym ⊡ A_sym)
    @test (@inferred frommandel(SymmetricTensor{2,dim}, tomandel(A_sym))) ≈ A_sym
    @test (@inferred frommandel(SymmetricTensor{4,dim}, tomandel(AA_sym))) ≈ AA_sym
    @test (@inferred tomandel(AA)) * (@inferred tomandel(A)) ≈ tomandel(AA ⊡ A)
    @test (@inferred frommandel(Tensor{2,dim}, tomandel(A))) ≈ A
    @test (@inferred frommandel(Tensor{4,dim}, tomandel(AA))) ≈ AA

    if T==Float64
        num_components = Int((dim^2+dim)/2)
        @test isa(fromvoigt(SymmetricTensor{2,dim}, rand(num_components) .> 0.5), SymmetricTensor{2,dim,Bool})
        @test isa(fromvoigt(SymmetricTensor{4,dim}, rand(num_components,num_components) .> 0.5), SymmetricTensor{4,dim,Bool})
    end
end
end # of testsection
end # of testsection
