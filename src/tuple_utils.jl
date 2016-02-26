# The code here is derived from the package ForwardDiff.jl

#ForwardDiff.jl is licensed under the MIT License:
#
#Copyright (c) 2015: Jarrett Revels, Theodore Papamarkou, Miles Lubin, and other contributors
#
#Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.


# This file contrains tuple operators that work uniformly on all elements of the tuple.

function tupexpr(f,N)
    ex = Expr(:tuple, [f(i) for i=1:N]...)
    return quote
        @inbounds return $ex
    end
end


function inline_body(T)
    if T == Float64 || T == Float32
        return Expr(:meta, :inline)
    else
        return Expr(:tuple)
    end
end

@generated function to_tuple{N}(::Type{NTuple{N}},  data::AbstractVecOrMat)
    return Expr(:tuple, [:(data[$i]) for i=1:N]...)
end

@inline to_tuple{N}(::Type{NTuple{N}},  data::NTuple{N}) = data


@generated function zero_tuple{N,T}(TT::Type{NTuple{N,T}})
    result = tupexpr((i) -> :z, N)
    return quote
        $(inline_body(T))
        z = zero(T)
        return $result
    end
end

@generated function const_tuple{N,T}(TT::Type{NTuple{N,T}}, v)
    result = tupexpr((i) -> :z, N)
    return quote
        $(inline_body(T))
        z = convert(T, v)
        return $result
    end
end

@generated function rand_tuple{N, T}(::Type{NTuple{N,T}})
    b = tupexpr((i) -> :(rand(T)), N)
      return quote
        $(inline_body(T))
        $b
    end
end


@generated function scale_tuple{N, T}(tup::NTuple{N, T}, x)
    b = tupexpr((i) -> :(tup[$i] * x), N)
      return quote
        $(inline_body(T))
        $b
    end
end

@generated function div_tuple_by_scalar{N, T}(tup::NTuple{N, T}, x)
    b = tupexpr((i) -> :(tup[$i] / x), N)
      return quote
        $(inline_body(T))
        $b
    end
end


@generated function minus_tuple{N, T}(tup::NTuple{N, T})
    b = tupexpr((i) -> :(-tup[$i]), N)
      return quote
        $(inline_body(T))
        $b
    end
end

@generated function subtract_tuples{N, T1, T2}(a::NTuple{N, T1}, b::NTuple{N, T2})
    b = tupexpr((i) -> :(a[$i] - b[$i]), N)
      return quote
        $(inline_body(promote_type(T1, T2)))
        $b
    end
end

@generated function add_tuples{N, T1, T2}(a::NTuple{N, T1}, b::NTuple{N, T2})
    b =  tupexpr((i) -> :(a[$i] + b[$i]), N)
    return quote
        $(inline_body(promote_type(T1, T2)))
        $b
    end
end

@generated function scalar_mul_tuples{N, T1, T2}(a::NTuple{N, T1}, b::NTuple{N, T2})
     b = tupexpr((i) -> :(a[$i] * b[$i]), N)
    return quote
        $(inline_body(promote_type(T1, T2)))
        $b
    end
end

@generated function scalar_div_tuples{N, T1, T2}(a::NTuple{N, T2}, b::NTuple{N, T1})
    b = tupexpr((i) -> :(a[$i] / b[$i]), N)
    return quote
        $(inline_body(promote_type(T1, T2)))
        $b
    end
end

