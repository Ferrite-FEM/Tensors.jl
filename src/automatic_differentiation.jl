using ForwardDiff
using ContMechTensors
import ForwardDiff: Dual, partials


######################
# Extraction methods #
######################

# Extractions are supposed to unpack the partials in a tensor
# and put it into a tensor of higher order.
# The extraction methods need to know the input type to the function
# that generated the result. The reason for this is that there is no
# difference in the output type (except the number of partials) for
# norm(v) and det(T) where v is a vector and T is a second order tensor.
# For dim = 1 the type is exactly the same.

# Scalar
@inline function _extract{N, T}(v::Dual{N, T}, ::Vec{N})
    Vec{N, T}(partials(v).values)
end

@inline function _extract{N, T, dim, T2}(v::Dual{N, T}, ::SymmetricTensor{2, dim, T2, N})
    SymmetricTensor{2, dim, T}(partials(v).values)
end

@inline function _extract{N, T, dim, T2}(v::Dual{N, T}, ::Tensor{2, dim, T2, N})
    Tensor{2, dim, T}(partials(v).values)
end

# Vec
@inline function _extract{D <: Dual}(v::Vec{1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1])
        d = Tensor{2, 1}((p1[1],))
    end
    return d
end

@inline function _extract{D <: Dual}(v::Vec{2, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1])
        p2 = partials(v[2])
        d = Tensor{2, 2}((p1[1], p2[1], p1[2], p2[2]))
    end
    return d
end

@inline function _extract{D <: Dual}(v::Vec{3, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1])
        p2 = partials(v[2])
        p3 = partials(v[3])
        d = Tensor{2, 3}((p1[1], p2[1], p3[1], p1[2], p2[2], p3[2], p1[3], p2[3], p3[3]))
    end
    return d
end

# Second order tensor
@inline function _extract{D <: Dual}(v::Tensor{2, 1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        d = Tensor{4, 1}((p1[1],))
    end
    return d
end

@inline function _extract{D <: Dual}(v::SymmetricTensor{2, 1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        d = SymmetricTensor{4, 1}((p1[1],))
    end
    return d
end

@inline function _extract{D <: Dual}(v::Tensor{2, 2, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        p2 = partials(v[2,1])
        p3 = partials(v[1,2])
        p4 = partials(v[2,2])
        d = Tensor{4, 2}((p1[1], p2[1], p3[1], p4[1],
                          p1[2], p2[2], p3[2], p4[2],
                          p1[3], p2[3], p3[3], p4[3],
                          p1[4], p2[4], p3[4], p4[4]))
    end
    return d
end

@inline function _extract{D <: Dual}(v::SymmetricTensor{2, 2, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        p2 = partials(v[2,1])
        p3 = partials(v[2,2])
        d = SymmetricTensor{4, 2}((p1[1], p2[1], p3[1],
                                   p1[2], p2[2], p3[2],
                                   p1[3], p2[3], p3[3]))
    end
    return d
end

@inline function _extract{D <: Dual}(v::Tensor{2, 3, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        p2 = partials(v[2,1])
        p3 = partials(v[3,1])
        p4 = partials(v[1,2])
        p5 = partials(v[2,2])
        p6 = partials(v[3,2])
        p7 = partials(v[1,3])
        p8 = partials(v[2,3])
        p9 = partials(v[3,3])
        d = Tensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1],
                          p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2],  #    ###  #
                          p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3], p9[3],  #    # #  #
                          p1[4], p2[4], p3[4], p4[4], p5[4], p6[4], p7[4], p8[4], p9[4],  ###  ###  ###
                          p1[5], p2[5], p3[5], p4[5], p5[5], p6[5], p7[5], p8[5], p9[5],
                          p1[6], p2[6], p3[6], p4[6], p5[6], p6[6], p7[6], p8[6], p9[6],
                          p1[7], p2[7], p3[7], p4[7], p5[7], p6[7], p7[7], p8[7], p9[7],
                          p1[8], p2[8], p3[8], p4[8], p5[8], p6[8], p7[8], p8[8], p9[8],
                          p1[9], p2[9], p3[9], p4[9], p5[9], p6[9], p7[9], p8[9], p9[9]))
    end
    return d
end

@inline function _extract{D <: Dual}(v::SymmetricTensor{2, 3, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        p2 = partials(v[2,1])
        p3 = partials(v[3,1])
        p4 = partials(v[2,2])
        p5 = partials(v[3,2])
        p6 = partials(v[3,3])

        d = SymmetricTensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1],
                                   p1[2], p2[2], p3[2], p4[2], p5[2], p6[2],
                                   p1[3], p2[3], p3[3], p4[3], p5[3], p6[3],
                                   p1[4], p2[4], p3[4], p4[4], p5[4], p6[4],
                                   p1[5], p2[5], p3[5], p4[5], p5[5], p6[5],
                                   p1[6], p2[6], p3[6], p4[6], p5[6], p6[6]))
    end
    return d
end

##################
# Load functions #
##################

# Loaders are supposed to take a tensor of real values and convert it
# into a tensor of dual values where the seeds are correctly defind.

@inline function _load{T}(v::Vec{1, T})
    @inbounds v_dual = Vec{1}((Dual(v[1], one(T)),))
    return v_dual
end

@inline function _load{T}(v::Vec{2, T})
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{2}((Dual(v[1], o, z),
                               Dual(v[2], z, o)))
    return v_dual
end

@inline function _load{T}(v::Vec{3, T})
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{3}((Dual(v[1], o, z, z),
                               Dual(v[2], z, o, z),
                               Dual(v[3], z, z, o)))
    return v_dual
end

# Second order tensors
@inline function _load{T}(v::Tensor{2, 1, T})
    @inbounds v_dual = Tensor{2, 1}((Dual(v.data.data[1], one(T)),))
    return v_dual
end

@inline function _load{T}(v::SymmetricTensor{2, 1, T})
    @inbounds v_dual = SymmetricTensor{2, 1}((Dual(v.data.data[1], one(T)),))
    return v_dual
end

@inline function _load{T}(v::Tensor{2, 2, T})
    data = v.data.data
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Tensor{2, 2}((Dual(data[1], o, z, z, z),
                                     Dual(data[2], z, o, z, z),
                                     Dual(data[3], z, z, o, z),
                                     Dual(data[4], z, z, z, o)))
    return v_dual
end

@inline function _load{T}(v::SymmetricTensor{2, 2, T})
    data = v.data.data
    o = one(T)
    o2 = T(1/2)
    z = zero(T)
    @inbounds v_dual = SymmetricTensor{2, 2}((Dual(data[1], o, z, z),
                                              Dual(data[2], z, o2, z),
                                              Dual(data[3], z, z, o)))
    return v_dual
end

@inline function _load{T}(v::Tensor{2, 3, T})
    data = v.data.data
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Tensor{2, 3}((Dual(data[1], o, z, z, z, z, z, z, z, z),
                                     Dual(data[2], z, o, z, z, z, z, z, z, z),
                                     Dual(data[3], z, z, o, z, z, z, z, z, z),
                                     Dual(data[4], z, z, z, o, z, z, z, z, z),
                                     Dual(data[5], z, z, z, z, o, z, z, z, z),
                                     Dual(data[6], z, z, z, z, z, o, z, z, z),
                                     Dual(data[7], z, z, z, z, z, z, o, z, z),
                                     Dual(data[8], z, z, z, z, z, z, z, o, z),
                                     Dual(data[9], z, z, z, z, z, z, z, z, o)))
    return v_dual
end

@inline function _load{T}(v::SymmetricTensor{2, 3, T})
    data = v.data.data
    o = one(T)
    o2 = T(1/2)
    z = zero(T)
    @inbounds v_dual = SymmetricTensor{2, 3}((Dual(data[1], o, z, z, z, z, z),
                                              Dual(data[2], z, o2, z, z, z, z),
                                              Dual(data[3], z, z, o2, z, z, z),
                                              Dual(data[4], z, z, z, o, z, z),
                                              Dual(data[5], z, z, z, z, o2, z),
                                              Dual(data[6], z, z, z, z, z, o)))
    return v_dual
end

function gradient{F}(f::F, v::Union{SecondOrderTensor, Vec})
    v_dual = _load(v)
    res = f(v_dual)
    return _extract(res, v)
end

function hessian{F}(f::F, v::Union{SecondOrderTensor, Vec})
    gradf = y -> gradient(f, y)
    return gradient(gradf, v)
end
