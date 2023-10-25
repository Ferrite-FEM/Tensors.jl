@testset "dotmacro" begin
    x, y = (rand(Tensor{2,3}) for _ in 1:2);
    v, u = (rand(Tensor{1,3}) for _ in 1:2);
    @test Tensors.m_dcontract(x, y) ≈ dcontract(x, y)
    @test Tensors.m_dot(x, y) ≈ dot(x, y)
    @test Tensors.m_otimes(u, v) ≈ otimes(u, v)
end