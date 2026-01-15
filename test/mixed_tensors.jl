using Test 
import Tensors: makemixed
@testset "MixedTensors" begin
    @testset "dcontract" begin
        ar = rand(Tensor{2,3})
        a = makemixed(ar)
        br = rand(Tensor{4,3})
        b = makemixed(br)
        @test dcontract(a,b) ≈ dcontract(ar, br)
        @test dcontract(b,a) ≈ dcontract(br, ar)
    end

    @testset "ad" begin
        v1 = rand(Vec{2})
        v2 = rand(Vec{2})
        v1_3d = Vec{3}((v1[1], v1[2], 0.0))
        v2_3d = Vec{3}((v2[1], v2[2], 0.0))
        foo(v::Vec{2}) = cross(v, v2)
        foo(v::Vec{3}) = cross(v, v2_3d)
        dfdv_2d = gradient(foo, v1)
        dfdv_3d = gradient(foo, v1_3d)  # Using regular Tensors 
        @test dfdv_2d ≈ dfdv_3d[:,1:2]

        ar = rand(Tensor{2,3})
        a = makemixed(ar)
        bar(x) = x ⋅ x
        barm(x) = makemixed(bar(x))
        @test gradient(bar, ar) ≈ gradient(bar, a)
        @test gradient(bar, ar) ≈ gradient(barm, a)
    end
end
