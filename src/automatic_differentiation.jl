import ForwardDiff: Dual, partials, value

######################
# Extraction methods #
######################

# Extractions are supposed to unpack the value and the partials
# The partials should be put into a tensor of higher order.
# The extraction methods need to know the input type to the function
# that generated the result. The reason for this is that there is no
# difference in the output type (except the number of partials) for
# norm(v) and det(T) where v is a vector and T is a second order tensor.
# For dim = 1 the type is exactly the same.

# Scalar
@inline function _extract_value{N, T}(v::Dual{N, T}, ::Vec{N})
    return value(v)
end
@inline function _extract_gradient{N, T}(v::Dual{N, T}, ::Vec{N})
    return Vec{N, T}(partials(v).values)
end

@inline function _extract_value{N, T, dim, T2}(v::Dual{N, T}, ::SymmetricTensor{2, dim, T2, N})
    return value(v)
end
@inline function _extract_gradient{N, T, dim, T2}(v::Dual{N, T}, ::SymmetricTensor{2, dim, T2, N})
    return SymmetricTensor{2, dim, T}(partials(v).values)
end

@inline function _extract_value{N, T, dim, T2}(v::Dual{N, T}, ::Tensor{2, dim, T2, N})
    return value(v)
end
@inline function _extract_gradient{N, T, dim, T2}(v::Dual{N, T}, ::Tensor{2, dim, T2, N})
    return Tensor{2, dim, T}(partials(v).values)
end

# Vec
@inline function _extract_value{D <: Dual}(v::Vec{1, D}, ::Any)
    @inbounds begin
        v1 = value(v[1])
        f = Vec{1}((v1,))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Vec{1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1])
        ∇f = Tensor{2, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::Vec{2, D}, ::Any)
    @inbounds begin
        v1, v2 = value(v[1]), value(v[2])
        f = Vec{2}((v1, v2))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Vec{2, D}, ::Any)
    @inbounds begin
        p1, p2 = partials(v[1]), partials(v[2])
        ∇f = Tensor{2, 2}((p1[1], p2[1], p1[2], p2[2]))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::Vec{3, D}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1]), value(v[2]), value(v[3])
        f = Vec{3}((v1, v2, v3))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Vec{3, D}, ::Any)
    @inbounds begin
        p1, p2, p3 = partials(v[1]), partials(v[2]), partials(v[3])
        ∇f = Tensor{2, 3}((p1[1], p2[1], p3[1], p1[2], p2[2], p3[2], p1[3], p2[3], p3[3]))
    end
    return ∇f
end

# Second order tensor
@inline function _extract_value{D <: Dual}(v::Tensor{2, 1, D}, ::Any)
    @inbounds begin
        v1 = value(v[1,1])
        f = Tensor{2, 1}((v1,))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Tensor{2, 1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = Tensor{4, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::SymmetricTensor{2, 1, D}, ::Any)
    @inbounds begin
        v1 = value(v[1,1])
        f = SymmetricTensor{2, 1}((v1,))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::SymmetricTensor{2, 1, D}, ::Any)
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = SymmetricTensor{4, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::Tensor{2, 2, D}, ::Any)
    @inbounds begin
        v1, v2, v3, v4 = value(v[1,1]), value(v[2,1]), value(v[1,2]), value(v[2,2])
        f = Tensor{2, 2}((v1, v2, v3, v4))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Tensor{2, 2, D}, ::Any)
    @inbounds begin
        p1, p2, p3, p4 = partials(v[1,1]), partials(v[2,1]), partials(v[1,2]), partials(v[2,2])
        ∇f = Tensor{4, 2}((p1[1], p2[1], p3[1], p4[1],
                           p1[2], p2[2], p3[2], p4[2],
                           p1[3], p2[3], p3[3], p4[3],
                           p1[4], p2[4], p3[4], p4[4]))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::SymmetricTensor{2, 2, D}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[2,2])
        f = SymmetricTensor{2, 2}((v1, v2, v3))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::SymmetricTensor{2, 2, D}, ::Any)
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[2,2])
        ∇f = SymmetricTensor{4, 2}((p1[1], p2[1], p3[1],
                                    p1[2], p2[2], p3[2],
                                    p1[3], p2[3], p3[3]))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::Tensor{2, 3, D}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[3,1])
        v4, v5, v6 = value(v[1,2]), value(v[2,2]), value(v[3,2])
        v7, v8, v9 = value(v[1,3]), value(v[2,3]), value(v[3,3])
        f = Tensor{2,3}((v1, v2, v3, v4, v5, v6, v7, v8, v9))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::Tensor{2, 3, D}, ::Any)
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[3,1])
        p4, p5, p6 = partials(v[1,2]), partials(v[2,2]), partials(v[3,2])
        p7, p8, p9 = partials(v[1,3]), partials(v[2,3]), partials(v[3,3])
        ∇f = Tensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1], p7[1], p8[1], p9[1],
                           p1[2], p2[2], p3[2], p4[2], p5[2], p6[2], p7[2], p8[2], p9[2],  #    ###  #
                           p1[3], p2[3], p3[3], p4[3], p5[3], p6[3], p7[3], p8[3], p9[3],  #    # #  #
                           p1[4], p2[4], p3[4], p4[4], p5[4], p6[4], p7[4], p8[4], p9[4],  ###  ###  ###
                           p1[5], p2[5], p3[5], p4[5], p5[5], p6[5], p7[5], p8[5], p9[5],
                           p1[6], p2[6], p3[6], p4[6], p5[6], p6[6], p7[6], p8[6], p9[6],
                           p1[7], p2[7], p3[7], p4[7], p5[7], p6[7], p7[7], p8[7], p9[7],
                           p1[8], p2[8], p3[8], p4[8], p5[8], p6[8], p7[8], p8[8], p9[8],
                           p1[9], p2[9], p3[9], p4[9], p5[9], p6[9], p7[9], p8[9], p9[9]))
    end
    return ∇f
end

@inline function _extract_value{D <: Dual}(v::SymmetricTensor{2, 3, D}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[3,1])
        v4, v5, v6 = value(v[2,2]), value(v[3,2]), value(v[3,3])
        f = SymmetricTensor{2, 3}((v1, v2, v3, v4, v5, v6))
    end
    return f
end
@inline function _extract_gradient{D <: Dual}(v::SymmetricTensor{2, 3, D}, ::Any)
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[3,1])
        p4, p5, p6 = partials(v[2,2]), partials(v[3,2]), partials(v[3,3])
        ∇f = SymmetricTensor{4, 3}((p1[1], p2[1], p3[1], p4[1], p5[1], p6[1],
                                    p1[2], p2[2], p3[2], p4[2], p5[2], p6[2],
                                    p1[3], p2[3], p3[3], p4[3], p5[3], p6[3],
                                    p1[4], p2[4], p3[4], p4[4], p5[4], p6[4],
                                    p1[5], p2[5], p3[5], p4[5], p5[5], p6[5],
                                    p1[6], p2[6], p3[6], p4[6], p5[6], p6[6]))
    end
    return ∇f
end

##################
# Load functions #
##################

# Loaders are supposed to take a tensor of real values and convert it
# into a tensor of dual values where the seeds are correctly defined.

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

"""
```julia
gradient(f::Function, v::Union{SecondOrderTensor, Vec})
gradient(f::Function, v::Union{SecondOrderTensor, Vec}, :all)
```
Computes the gradient of the input function. If the (pseudo)-keyword `all`
is given, the value of the function is also returned as a second output argument.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇f = gradient(norm, A)
2×2 ContMechTensors.SymmetricTensor{2,2,Float64,3}:
 0.434906  0.56442
 0.56442   0.416793

julia> ∇f, f = gradient(norm, A, :all);
```
"""
function Base.gradient{F}(f::F, v::Union{SecondOrderTensor, Vec})
    v_dual = _load(v)
    res = f(v_dual)
    return _extract_gradient(res, v)
end
function Base.gradient{F}(f::F, v::Union{SecondOrderTensor, Vec}, ::Symbol)
    v_dual = _load(v)
    res = f(v_dual)
    return _extract_gradient(res, v), _extract_value(res, v)
end

"""
```julia
hessian(f::Function, v::Union{SecondOrderTensor, Vec})
hessian(f::Function, v::Union{SecondOrderTensor, Vec}, :all)
```
Computes the hessian of the input function. If the (pseudo)-keyword `all`
is given, the lower order results (gradient and value) of the function is
also returned as a second and third output argument.

**Example:**

```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇∇f = hessian(norm, A)
2×2×2×2 ContMechTensors.SymmetricTensor{4,2,Float64,9}:
[:, :, 1, 1] =
  0.596851  -0.180684
 -0.180684  -0.133425

[:, :, 2, 1] =
 -0.180684   0.133546
  0.133546  -0.173159

[:, :, 1, 2] =
 -0.180684   0.133546
  0.133546  -0.173159

[:, :, 2, 2] =
 -0.133425  -0.173159
 -0.173159   0.608207

julia> ∇∇f, ∇f, f = hessian(norm, A, :all);
```
"""
function hessian{F}(f::F, v::Union{SecondOrderTensor, Vec})
    gradf = y -> gradient(f, y)
    return gradient(gradf, v)
end

function hessian{F}(f::F, v::Union{SecondOrderTensor, Vec}, ::Symbol)
    gradf = y -> gradient(f, y)
    return gradient(gradf, v), gradient(f, v, :all)...
end
