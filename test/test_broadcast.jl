@testset "broadcast lattice" begin
    S = rand(Tensor{2, 2}); Ss = rand(SymmetricTensor{2, 2}); v = rand(Vec{3})
    @test (S .+ 1.0) isa Tensor{2, 2, Float64}
    @test Array(S .+ 1.0) == Array(S) .+ 1.0
    @test (Ss .* 2) isa Tensor{2, 2, Float64}      # general f: densified
    @test (S .+ Ss) isa Tensor{2, 2, Float64}
    @test sqrt.(v) isa Vec{3, Float64}
    @test (S .+ rand(2, 2)) isa Matrix{Float64}    # ordinary arrays win
    @test (v .+ rand(Tensor{2, 3})) isa Matrix{Float64}  # shape mismatch -> array semantics
    @test ((x -> "s").(v)) isa Vector{String}      # non-Number eltype -> Array
    m = rand(MixedTensor{2, Tuple{2, 3}})
    @test (m .* 2) isa MixedTensor{2, Tuple{2, 3}, Float64}
    @test (@inferred (S .* 2.0)) isa Tensor{2, 2, Float64}
end
