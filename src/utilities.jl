#####################
# Utility functions #
#####################

#######################################################################################
# See :https://groups.google.com/forum/?nomobile=true#!topic/julia-users/x8Z5Vrq53RM
macro gen_code(f)
  isa(f, Expr) || error("gen_code macro must precede a function definition")
  (f.head == :function) || error("gen_code macro must precede a function definition")
  unshift!(f.args[2].args, :(code = :nothing))
  push!(f.args[2].args, :(code))

  e = :(@generated $f)
  return Expr(:escape, e)
end

function __append_code(a, b)
  return :($a; $b)
end

macro code(e)
  isa(e, Expr) || error("can't generate code from non-expressions")
  return Expr(:escape, :(code = __append_code(code, $e)))
end

#######################################################################################
tensor_create{order, dim}(::Type{Tensor{order, dim}}, f) = tensor_create(Tensor{order, dim, Float64}, f)
function tensor_create{order, dim, T}(::Type{Tensor{order, dim, T}}, f)
    if order == 1
        ex = Expr(:tuple, [f(i) for i=1:dim]...)
    elseif order == 2
        ex = Expr(:tuple, [f(i,j) for i=1:dim, j=1:dim]...)
    elseif order == 4
        ex = Expr(:tuple, [f(i,j,k,l) for i=1:dim, j=1:dim, k = 1:dim, l = 1:dim]...)
    end
    return :($ex)
end

tensor_create{order, dim}(::Type{SymmetricTensor{order, dim}}, f) = tensor_create(SymmetricTensor{order, dim, Float64}, f)
function tensor_create{order, dim, T}(::Type{SymmetricTensor{order, dim, T}}, f)
    ex = Expr(:tuple)
    if order == 2
        for i in 1:dim, j in i:dim
            push!(ex.args, f(j, i))
        end
    elseif order == 4
        for k in 1:dim, l in k:dim, i in 1:dim, j in i:dim
            push!(ex.args, f(j, i, l, k))
        end
    end
    return :($ex)
end

@generated function to_tuple{N}(::Type{NTuple{N}},  data::AbstractArray)
    return Expr(:tuple, [:(data[$i]) for i=1:N]...)
end

@inline to_tuple{N}(::Type{NTuple{N}},  data::NTuple{N}) = data
