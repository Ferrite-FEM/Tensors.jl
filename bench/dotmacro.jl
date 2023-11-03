module DotTensorModule
    import Tensors: Tensors, Tensor, SymmetricTensor
    
    function getrange(expr)
        @assert expr.head === :call
        @assert expr.args[1] === :(:)
        @assert all(x->isa(x,Number), expr.args[2:end])
        @assert length(expr.args) ∈ (3,4)
        if length(expr.args) == 3 # from:to range
            return expr.args[2]:expr.args[3]
        else #length(expr.args) == 4 # from:step:to range
            return expr.args[2]:expr.args[3]:expr.args[4]
        end
    end

    function getiterable(expr)
        if expr.head === :call && expr.args[1] === :(:)
            return getrange(expr)
        elseif expr.head === :(tuple)
            return (a for a in expr.args)
        else
            error("Don't know what to do with $(expr.head)")
        end
    end

    function loop_over_cases(loopsym, cases, expr)
        exprs = Expr(:tuple)
        for loopvar in getiterable(cases)
            tmpexpr = deepcopy(expr)
            f(s::Symbol) = (s === loopsym ? loopvar : s)
            Tensors.replace_args!(f, tmpexpr.args)
            push!(exprs.args, esc(tmpexpr))
        end
        return exprs
    end

    function foreach(expr)
        @assert expr.head === :for
        loopsym = expr.args[1].args[1]
        isa(loopsym, Symbol) || error("Can only loop over one variable")
        cases = expr.args[1].args[2]
        codeblock = expr.args[2]::Expr
        @assert codeblock.head === :block
        return loop_over_cases(loopsym, cases, codeblock)
    end

    macro foreach(expr)
        return foreach(expr)
    end

    @foreach for dim in 1:3
        @foreach for TT in (Tensor, SymmetricTensor)
            Tensors.@tensor_product(@inline @inbounds function dot(A::TT{2,dim}, B::TT{2,dim})
                C[i,j] = A[i,k]*B[k,j]
            end)

            Tensors.@tensor_product(@inline @inbounds function dcontract(A::TT{2,dim}, B::TT{2,dim})
                C = A[i,j]*B[i,j]
            end, muladd)
            Tensors.@tensor_product(@inline @inbounds function dcontract(A::TT{4,dim}, B::TT{2,dim})
                C[i,j] = A[i,j,k,l]*B[k,l]
            end, muladd)
            Tensors.@tensor_product(@inline @inbounds function dcontract(A::TT{2,dim}, B::TT{4,dim})
                C[k,l] = A[i,j]*B[i,j,k,l]
            end, muladd)
            Tensors.@tensor_product(@inline @inbounds function dcontract(A::TT{4,dim}, B::TT{4,dim})
                C[i,j,k,l] = A[i,j,m,n]*B[m,n,k,l]
            end, muladd)

            Tensors.@tensor_product(@inline @inbounds function otimes(A::TT{2,dim}, B::TT{2,dim})
                C[i,j,k,l] = A[i,j]*B[k,l]
            end)
        end
        Tensors.@tensor_product(@inline @inbounds function dot(A::Tensor{1,dim}, B::Tensor{1,dim})
            C = A[i]*B[i]
        end, muladd)
        Tensors.@tensor_product(@inline @inbounds function dot(A::Tensor{2,dim}, B::Tensor{1,dim})
            C[i] = A[i,j]*B[j]
        end, muladd)
        Tensors.@tensor_product(@inline @inbounds function dot(A::Tensor{1,dim}, B::Tensor{2,dim})
            C[j] = A[i]*B[i,j]
        end, muladd)
        Tensors.@tensor_product(@inline @inbounds function otimes(A::Tensor{1,dim}, B::Tensor{1,dim})
            C[i,j] = A[i]*B[j]
        end)
    end
end

using BenchmarkTools, StaticArrays, DataFrames, Test
using Tensors
import .DotTensorModule
import .DotTensorModule as DTM

# Relevant non-simd number type to check performance
const DT{T,N} = Tensors.Dual{Nothing, T, N}

macro each(expr, nmax=100)
    expr = esc(expr)
    nmax = esc(nmax)
    return quote
        ($expr for _ in 1:$nmax)
    end
end

function run_benchmarks(T=DT{Float64,6}; M=Tensors)
    _dot = M.dot
    _dcontract = M.dcontract
    _otimes = M.otimes
    dot22, ddot22, otim22 = @each Float64[]
    dot22s, ddot22s, otim22s = @each Float64[]
    ddot44 = Float64[]
    ddot44s = Float64[]
    otim11 = Float64[]
    dims = collect(1:3)
    for dim in dims
        @show dim
        u, v   = @each rand(Vec{dim,T})
        a, b   = @each rand(Tensor{2,dim,T})
        as, bs = @each rand(SymmetricTensor{2,dim,T})
        A, B   = @each rand(Tensor{4,dim,T})
        As, Bs = @each rand(SymmetricTensor{4,dim,T})

        push!(dot22, @belapsed $_dot($a, $b))
        push!(ddot22, @belapsed $_dcontract($a, $b))
        push!(otim22, @belapsed $_otimes($a, $b))
        push!(dot22s, @belapsed $_dot($as, $bs))
        push!(ddot22s, @belapsed $_dcontract($as, $bs))
        push!(otim22s, @belapsed $_otimes($as, $bs))

        push!(ddot44, @belapsed $_dcontract($A, $B))
        push!(ddot44s, @belapsed $_dcontract($As, $Bs))
        
        push!(otim11, @belapsed $_otimes($u, $v))

        if M===Tensors
            println("Running tests")
            @testset "Check" begin
                @test Tensors.dot(a, b) ≈ DTM.dot(a, b)
                @test Tensors.dcontract(a, b) ≈ DTM.dcontract(a, b)
                @test Tensors.otimes(a, b) ≈ DTM.otimes(a, b)

                @test Tensors.dot(as, bs) ≈ DTM.dot(as, bs)
                @test Tensors.dcontract(as, bs) ≈ DTM.dcontract(as, bs)
                @test Tensors.otimes(as, bs) ≈ DTM.otimes(as, bs)
                
                @test Tensors.dcontract(A, B) ≈ DTM.dcontract(A, B)
                @test Tensors.dcontract(As, Bs) ≈ DTM.dcontract(As, Bs)

                @test Tensors.otimes(u, v) ≈ DTM.otimes(u, v)
            end
        end
    end
    dims = dims
    df = DataFrame((key => (key==:dim ? t : t*1e9) for (key,t) in 
        ((:dim, dims), (:dot22, dot22), (:dot22s, dot22s), 
        (:ddot22, ddot22), (:ddot22s, ddot22s), (:ddot44, ddot44), 
        (:ddot44s, ddot44s), (:otim11, otim11), (:otim22, otim22), 
        (:otim22s, otim22s)))...);
    return return df
end

df_tensors = run_benchmarks(;M=Tensors)
df_dotmacr = run_benchmarks(;M=DotTensorModule)

df = DataFrame((key=>key=="dim" ? df_tensors[!, key] : 
    (a=df_tensors[!, key]; b=df_dotmacr[!, key]; map((x,y)->(x,y), a, b)) for key in names(df_tensors))...)

df
# DotTensorModule
#= 
3×10 DataFrame
 Row │ dim    dot22               dot22s              ddot22              ddot22s            ddot44              ddot44s             otim11              otim22              otim22s                
     │ Int64  Tuple…              Tuple…              Tuple…              Tuple…             Tuple…              Tuple…              Tuple…              Tuple…              Tuple…                 
─────┼──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────    
   1 │     1  (1.9, 1.9)          (1.6, 2.1)          (1.6, 2.1)          (1.8, 1.9)         (1.9, 1.8)          (1.6, 2.1)          (1.8, 1.8)          (1.8, 1.8)          (1.6, 2.1)
   2 │     2  (10.3103, 11.5115)  (10.8108, 12.4)     (5.4, 7.60761)      (5.60561, 4.9049)  (76.2887, 87.1369)  (38.5081, 42.7273)  (4.7, 5.1)          (29.6891, 32.998)   (11.6116, 12.7127)     
   3 │     3  (39.3145, 43.7374)  (40.7669, 45.0505)  (12.7127, 11.4114)  (12.6126, 7.0)     (2433.33, 2377.78)  (490.256, 478.571)  (13.3133, 12.9129)  (284.262, 313.061)  (66.6328, 65.0357) 
=#
