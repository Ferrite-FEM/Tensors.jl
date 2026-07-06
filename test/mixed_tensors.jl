@testsection "MixedTensors" begin
    @testsection "regular_conversions" begin
        @test Tensors.isregular(MixedTensor{1, Tuple{3}}) # order 1 always regular
        for order in 2:4
            dims = rand(1:2, order)
            dims[1] = 3 # Make sure not regular
            TM = MixedTensor{order, Tuple{(dims...)}}
            tm = rand(TM)
            @test !Tensors.isregular(TM)
            @test (@inferred Tensors.regular_if_possible(TM)) === TM
            @test (@inferred Tensors.regular_if_possible(TM{Float32})) === TM{Float32}
            @test (@inferred Tensors.regular_if_possible(typeof(tm))) === typeof(tm)
            @test (@inferred Tensors.regular_if_possible(tm)) === tm

            TR = MixedTensor{order, Tuple{(ntuple(_->2, order)...)}}
            tr = rand(TR)
            TT = Tensor{order, 2}
            tt = TT((args...) -> getindex(tr, args...))
            @test Tensors.isregular(TR)
            @test (@inferred Tensors.regular_if_possible(TR)) === TT
            @test (@inferred Tensors.regular_if_possible(TR{Float32})) === TT{Float32}
            @test (@inferred Tensors.regular_if_possible(typeof(tr))) === typeof(tt)
            @test (@inferred Tensors.regular_if_possible(tr)) === tt

            @test (@inferred Tensors.makemixed(tr))::TR === tr
            @test (@inferred Tensors.makemixed(tt))::TR === tr
        end
    end
    @testsection "construction" begin
        @testsection "from tuples" begin
            data = ntuple(_ -> rand(), 6)
            TD = ForwardDiff.Dual{Nothing, Float64, 3}
            TB = MixedTensor2{2, 3}
            # Standard case
            @test isa(TB(data), MixedTensor2{2, 3, Float64, 6})
            # Always respecting the tensor's data type specification
            @test isa(TB{Float32}(data), MixedTensor2{2, 3, Float32, 6})
            @test isa(TB{TD}(data), MixedTensor2{2, 3, TD, 6})
            # Heterogeneous tuple
            @test isa(TB((1.f0, 1.0, 1, zero(TD), 2, 3)), MixedTensor2{2, 3, TD, 6})
        end
        @testsection "from function" begin
            T2 = MixedTensor{2, Tuple{(rand(1:3, 2)...)}}
            T3 = MixedTensor{3, Tuple{(rand(1:3, 3)...)}}
            T4 = MixedTensor{4, Tuple{(rand(1:3, 4)...)}}
            m2, m3, m4 = rand.((T2, T3, T4))
            @test T2((i, j) -> m2[i, j]) == m2
            @test T3((i, j, k) -> m3[i, j, k]) == m3
            @test T4((i, j, k, l) -> m4[i, j, k, l]) == m4 
        end
        @testsection "from slice" begin
            v = rand(Vec{2})
            t = rand(Tensor{2,2})
            m = MixedTensor2{2, 3}((v.data..., t.data...))
            @test isa(m[:, 1], Vec{2})
            @test isa(m[2, :], Vec{3})
            @test (@inferred m[:,1]) == v
            @test (@inferred m[1,:]) == Vec((v[1], t[1,1], t[1,2]))
            @test m[:, 2] == t[:, 1]
            @test m[2, :] == Vec((v[2], t[2,1], t[2,2]))
        end
    end
    @testsection "basic ops" begin
        m23 = rand(MixedTensor2{2, 3})::MixedTensor2{2, 3}
        m21 = rand(MixedTensor2{2, 1})::MixedTensor2{2, 1}
        m231 = rand(MixedTensor3{2, 3, 1})::MixedTensor3{2, 3, 1}
        m2332 = rand(MixedTensor4{2, 3, 3, 2})::MixedTensor4{2, 3, 3, 2}
        m2131 = rand(MixedTensor4{2, 1, 3, 1})::MixedTensor4{2, 1, 3, 1}
        @testsection "transpose" begin
            @test ((@inferred transpose(m23))::MixedTensor2{3, 2})[2, 1] == m23[1, 2]
            @test ((@inferred transpose(m21))::MixedTensor2{1, 2})[1, 2] == m21[2, 1]
            @test transpose(m23) == m23'
            @test ((@inferred majortranspose(m2332))::MixedTensor4{3, 2, 2, 3})[3, 2, 2, 1] == m2332[2, 1, 3, 2]
            @test ((@inferred minortranspose(m2131))::MixedTensor4{1, 2, 1, 3})[1, 2, 1, 3] == m2131[2, 1, 3, 1]
        end
        @testsection "norm" begin
            for order in (1, 2, 3, 4)
                t = rand(MixedTensor{order, Tuple{(rand(1:3, order)...)}})
                @test norm(t) ≈ norm(collect(t.data))
            end
        end
    end
    @testsection "basic arithmetric" begin
        for T in (Float32, Float64, ForwardDiff.Dual{Nothing, Float64, 2})
            m23_f64 = rand(MixedTensor2{3, 2})
            m23_T = rand(MixedTensor2{3, 2, T})
            v = rand(T)
            for op in (+, -)
                s = (@inferred op(m23_f64, m23_T))
                @test isa(s, MixedTensor2{3, 2, promote_type(Float64, T)})
                @test all(s.data .≈ op.(m23_f64.data, m23_T.data))
            end
            for op in (*, /)
                s = (@inferred op(m23_f64, v))
                @test isa(s, MixedTensor2{3, 2, promote_type(Float64, T)})
                @test all(s.data .≈ op.(m23_f64.data, v))
            end
        end
    end
    @testsection "dot" begin
        function dot_index(a::MixedTensor2{di, ds}, v::Union{Vec{ds}, MixedTensor{1, Tuple{ds}}}) where {di, ds}
            return Vec{di}(i -> sum(a[i, j] * v[j] for j in 1:ds))
        end
        function dot_index(a::MixedTensor2{di, ds}, b::MixedTensor2{ds, dj}) where {di, ds, dj}
            return MixedTensor2{di, dj}((i, j) -> sum(a[i, s] * b[s, j] for s in 1:ds))
        end
        function dot_index(a::MixedTensor4{di, dj, dk, ds}, b::MixedTensor2{ds, dl}) where {di, dj, dk, ds, dl}
            return MixedTensor4{di, dj, dk, dl}((i, j, k, l) -> sum(a[i, j, k, s] * b[s, l] for s in 1:ds))
        end
        
        m2 = rand(MixedTensor{1, Tuple{2}})::MixedTensor{1}
        v = Vec(m2.data)
        m23 = rand(MixedTensor2{2, 3})
        m32 = rand(MixedTensor2{3, 2})
        m1232 = rand(MixedTensor4{1, 2, 3, 2})

        m23_dot_m32 = @inferred m23 ⋅ m32
        @test isa(m23_dot_m32, Tensor{2, 2})
        @test m23_dot_m32 ≈ Tensors.regular_if_possible(dot_index(m23, m32))
        @test m32 ⋅ m23 ≈ Tensors.regular_if_possible(dot_index(m32, m23))
        @test m1232 ⋅ m23 ≈ dot_index(m1232, m23)
        @test (@inferred m32 ⋅ m2)::Vec{3} ≈ dot_index(m32, m2)
        @test (@inferred m32 ⋅ v)::Vec{3} ≈ m32 ⋅ m2
    end

    @testsection "dcontract" begin
        function dcontract_index(a::MixedTensor2{di, dj}, b::MixedTensor2{di, dj}) where {di, dj}
            return sum(a.data[i] * b.data[i] for i in 1:(di*dj))
        end
        function dcontract_index(a::MixedTensor3{di, dj, dk}, b::MixedTensor3{dj, dk, dl}) where {di, dj, dk, dl}
            return MixedTensor2{di, dl}((i, l) -> sum(sum(a[i, j, k] * b[j, k, l] for j in 1:dj) for k in 1:dk))
        end
        function dcontract_index(a::MixedTensor4{di, dj, dk, dl}, b::MixedTensor2{dk, dl}) where {di, dj, dk, dl}
            return MixedTensor2{di, dj}((i, j) -> sum(sum(a[i, j, k, l] * b[k, l] for k in 1:dk) for l in 1:dl))
        end
        function dcontract_index(a::MixedTensor4{di, dj, dk, dk}, b::Tensor{2, dk}) where {di, dj, dk}
            return MixedTensor2{di, dj}((i, j) -> sum(sum(a[i, j, k, l] * b[k, l] for k in 1:dk) for l in 1:dk))
        end
        m31_1 = rand(MixedTensor2{3, 1})
        m31_2 = rand(MixedTensor2{3, 1})
        m32 = rand(MixedTensor2{3, 2})
        m213 = rand(MixedTensor3{2, 1, 3})
        m133 = rand(MixedTensor3{1, 3, 3})
        m3231 = rand(MixedTensor4{3, 2, 3, 1})

        m31_dd_m31 = @inferred m31_1 ⊡ m31_2
        @test isa(m31_dd_m31, Float64)
        @test m31_dd_m31 ≈ dcontract_index(m31_1, m31_2)
        @test m213 ⊡ m133 ≈ dcontract_index(m213, m133)
        @test m3231 ⊡ m31_1 ≈ dcontract_index(m3231, m31_1)
        @test majortranspose(m3231) ⊡ m32 ≈ dcontract_index(majortranspose(m3231), m32)

        # Correct conversion to `Tensor`
        @test isa(rand(MixedTensor4{2, 2, 1, 3}) ⊡ rand(MixedTensor2{1, 3}), Tensor{2, 2})
        # Multiplying mixed with regular
        m1322 = rand(MixedTensor4{1, 3, 2, 2})
        m3322 = rand(MixedTensor4{3, 3, 2, 2})
        t22 = rand(Tensor{2, 2})
        @test (@inferred m1322 ⊡ t22) ≈ dcontract_index(m1322, t22)
        @test ((@inferred m3322 ⊡ t22)::Tensor{2, 3}) ≈ Tensors.regular_if_possible(dcontract_index(m3322, t22))
    end

    @testsection "otimes" begin
        otimes_index(u::Vec{di}, v::Vec{dj}) where {di, dj} = MixedTensor2{di, dj}((i,j) -> u[i] * v[j])
        otimes_index(u::MixedTensor2{di, dj}, v::Vec{dk}) where {di, dj, dk} = MixedTensor3{di, dj, dk}((i,j,k) -> u[i, j] * v[k])
        otimes_index(u::Vec{di}, v::MixedTensor2{dj, dk}) where {di, dj, dk} = MixedTensor3{di, dj, dk}((i,j,k) -> u[i] * v[j, k])
        otimes_index(u::MixedTensor2{di, dj}, v::MixedTensor2{dk, dl}) where {di, dj, dk, dl} = MixedTensor4{di, dj, dk, dl}((i, j, k, l) -> u[i, j] * v[k, l])
        
        v2 = rand(Vec{2})
        v3 = rand(Vec{3})
        m23 = rand(MixedTensor2{2, 3})
        m31 = rand(MixedTensor2{3, 1})
        @test v2 ⊗ v3 ≈ otimes_index(v2, v3)
        @test m23 ⊗ v2 ≈ otimes_index(m23, v2)
        @test v3 ⊗ m31 ≈ otimes_index(v3, m31)
        @test m31 ⊗ m23 ≈ otimes_index(m31, m23)
    end

    @testsection "pinv" begin
        m22 = rand(Tensor{2,2})
        m33 = rand(Tensor{2,3})
        m23 = rand(MixedTensor2{2, 3})
        m32 = rand(MixedTensor2{3, 2})
        m13 = rand(MixedTensor2{1, 3})
        m31 = rand(MixedTensor2{3, 1})
        @test (@inferred pinv(m22)) ≈ pinv(Array(m22))
        @test (@inferred pinv(m33)) ≈ pinv(Array(m33))
        @test (@inferred pinv(m32)) ≈ pinv(Array(m32))
        @test (@inferred pinv(m31)) ≈ pinv(Array(m31))
        @test (@inferred pinv(m23)) ≈ pinv(Array(m23))
        @test (@inferred pinv(m13)) ≈ pinv(Array(m13))
    end

    @testsection "Automatic Differentation" begin
        # As code is mostly shared with regular `Tensor`s, the main part 
        # here is to test the extraction and insertion parts in the right locations. 
        # So we make some artifical tests but don't need too much nested functions.
        compmul(t, s) = sum(tv * sv for (tv, sv) in zip(t.data, s.data)) # d(compmul(t,s))/dt = s
        for order in 1:4  
            t = rand(MixedTensor{order, Tuple{(rand(1:3, order)...)}})
            s = rand(typeof(t))
            @test gradient(x -> compmul(x, s), t) ≈ s
            @test gradient(norm, t) ≈ t / norm(t)
        end

        # Also test derivatives when output has different dimension from input
        foo(v::Vec{3}) = Vec((v[1] + v[2], v[3] * v[1]))
        v = rand(Vec{3})
        dfdx = MixedTensor2{2, 3}((1.0, v[3], 1.0, 0.0, 0.0, v[1]))
        @test gradient(foo, v) ≈ dfdx
    end
    @testsection "errors" begin
        m32 = rand(MixedTensor2{2,3})
        @test_throws ArgumentError m32^2
    end
end
