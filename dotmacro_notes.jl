using Tensors
import Tensors: IndSyms, find_index # new internals used here

# For learning/seeing output
function get_string_expression(ci::IndSyms, ai::IndSyms, bi::IndSyms, dims::NamedTuple)
    si = tuple(sort(intersect(ai, bi))...)
    for cinds in Iterators.ProductIterator(tuple((1:dims[k] for k in ci)...))
        print("C[", join(string.(cinds), ", "), "] = ")
        parts = String[]
        for sinds in Iterators.ProductIterator(tuple((1:dims[k] for k in si)...))
            ainds = tuple((find_index(a, ci, cinds, si, sinds) for a in ai)...)
            binds = tuple((find_index(b, ci, cinds, si, sinds) for b in bi)...)
            push!(parts, string("A[", 
                join(string.(ainds), ", "), "] * B[",
                join(string.(binds), ", "), "]"))
        end
        println(join(parts, " + "))
    end
end
