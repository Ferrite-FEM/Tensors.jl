#################
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
#########################################################################################

function is_diagonal_index(dim::Int, i::Int)
    if dim == 1
        return true
    elseif dim == 2
        return i in (1,3)
    else
        return i in (1,4,6)
    end
end
