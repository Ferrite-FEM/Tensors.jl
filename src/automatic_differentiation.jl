import ForwardDiff: Dual, partials, value

@static if isdefined(LinearAlgebra, :gradient)
    import LinearAlgebra.gradient
end

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
@inline function _extract_value(v::Dual, ::Vec)
    return value(v)
end
@inline function _extract_gradient(v::Dual, ::Vec{N}) where {N}
    return Vec{N}(partials(v).values)
end

@inline function _extract_value(v::Dual, ::SecondOrderTensor)
    return value(v)
end
@inline function _extract_gradient(v::Dual, ::SymmetricTensor{2, dim}) where {dim}
    return SymmetricTensor{2, dim}(partials(v).values)
end
@inline function _extract_gradient(v::Dual, ::Tensor{2, dim}) where {dim}
    return Tensor{2, dim}(partials(v).values)
end

# Vec
@inline function _extract_value(v::Vec{1, <: Dual}, ::Any)
    @inbounds begin
        v1 = value(v[1])
        f = Vec{1}((v1,))
    end
    return f
end
@inline function _extract_gradient(v::Vec{1, <: Dual}, ::Vec{1})
    @inbounds begin
        p1 = partials(v[1])
        ∇f = Tensor{2, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value(v::Vec{2, <: Dual}, ::Any)
    @inbounds begin
        v1, v2 = value(v[1]), value(v[2])
        f = Vec{2}((v1, v2))
    end
    return f
end
@inline function _extract_gradient(v::Vec{2, <: Dual}, ::Vec{2})
    @inbounds begin
        p1, p2 = partials(v[1]), partials(v[2])
        ∇f = Tensor{2, 2}((p1[1], p2[1], p1[2], p2[2]))
    end
    return ∇f
end

@inline function _extract_value(v::Vec{3, <: Dual}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1]), value(v[2]), value(v[3])
        f = Vec{3}((v1, v2, v3))
    end
    return f
end
@inline function _extract_gradient(v::Vec{3, <: Dual}, ::Vec{3})
    @inbounds begin
        p1, p2, p3 = partials(v[1]), partials(v[2]), partials(v[3])
        ∇f = Tensor{2, 3}((p1[1], p2[1], p3[1], p1[2], p2[2], p3[2], p1[3], p2[3], p3[3]))
    end
    return ∇f
end

# Second order tensor
@inline function _extract_value(v::Tensor{2, 1, <: Dual}, ::Any)
    @inbounds begin
        v1 = value(v[1,1])
        f = Tensor{2, 1}((v1,))
    end
    return f
end
@inline function _extract_gradient(v::Tensor{2, 1, <: Dual}, ::Tensor{2, 1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = Tensor{4, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value(v::SymmetricTensor{2, 1, <: Dual}, ::Any)
    @inbounds begin
        v1 = value(v[1,1])
        f = SymmetricTensor{2, 1}((v1,))
    end
    return f
end
@inline function _extract_gradient(v::SymmetricTensor{2, 1, <: Dual}, ::SymmetricTensor{2, 1})
    @inbounds begin
        p1 = partials(v[1,1])
        ∇f = SymmetricTensor{4, 1}((p1[1],))
    end
    return ∇f
end

@inline function _extract_value(v::Tensor{2, 2, <: Dual}, ::Any)
    @inbounds begin
        v1, v2, v3, v4 = value(v[1,1]), value(v[2,1]), value(v[1,2]), value(v[2,2])
        f = Tensor{2, 2}((v1, v2, v3, v4))
    end
    return f
end
@inline function _extract_gradient(v::Tensor{2, 2, <: Dual}, ::Tensor{2, 2})
    @inbounds begin
        p1, p2, p3, p4 = partials(v[1,1]), partials(v[2,1]), partials(v[1,2]), partials(v[2,2])
        ∇f = Tensor{4, 2}((p1[1], p2[1], p3[1], p4[1],
                           p1[2], p2[2], p3[2], p4[2],
                           p1[3], p2[3], p3[3], p4[3],
                           p1[4], p2[4], p3[4], p4[4]))
    end
    return ∇f
end

@inline function _extract_value(v::SymmetricTensor{2, 2, <: Dual}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[2,2])
        f = SymmetricTensor{2, 2}((v1, v2, v3))
    end
    return f
end
@inline function _extract_gradient(v::SymmetricTensor{2, 2, <: Dual}, ::SymmetricTensor{2, 2})
    @inbounds begin
        p1, p2, p3 = partials(v[1,1]), partials(v[2,1]), partials(v[2,2])
        ∇f = SymmetricTensor{4, 2}((p1[1], p2[1], p3[1],
                                    p1[2], p2[2], p3[2],
                                    p1[3], p2[3], p3[3]))
    end
    return ∇f
end

@inline function _extract_value(v::Tensor{2, 3, <: Dual}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[3,1])
        v4, v5, v6 = value(v[1,2]), value(v[2,2]), value(v[3,2])
        v7, v8, v9 = value(v[1,3]), value(v[2,3]), value(v[3,3])
        f = Tensor{2,3}((v1, v2, v3, v4, v5, v6, v7, v8, v9))
    end
    return f
end
@inline function _extract_gradient(v::Tensor{2, 3, <: Dual}, ::Tensor{2, 3})
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

@inline function _extract_value(v::SymmetricTensor{2, 3, <: Dual}, ::Any)
    @inbounds begin
        v1, v2, v3 = value(v[1,1]), value(v[2,1]), value(v[3,1])
        v4, v5, v6 = value(v[2,2]), value(v[3,2]), value(v[3,3])
        f = SymmetricTensor{2, 3}((v1, v2, v3, v4, v5, v6))
    end
    return f
end
@inline function _extract_gradient(v::SymmetricTensor{2, 3, <: Dual}, ::SymmetricTensor{2, 3})
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

# for non dual variable
@inline function _extract_value(v::Any, ::Any)
    return v
end
for TensorType in (Tensor, SymmetricTensor)
    @eval begin
        @inline function _extract_gradient(v::T, x::$TensorType{order, dim}) where {T<:Real, order, dim}
            zero($TensorType{order, dim, T})
        end
        @generated function _extract_gradient(v::$TensorType{order, dim, T}, ::$TensorType{order, dim}) where {T<:Real, order, dim}
            RetType = $TensorType{order+order, dim, T}
            return quote
                $(Expr(:meta, :inline))
                zero($RetType)
            end
        end
    end
end

##################
# Load functions #
##################

# Loaders are supposed to take a tensor of real values and convert it
# into a tensor of dual values where the seeds are correctly defined.

@inline function _load(v::Vec{1, T}) where {T}
    @inbounds v_dual = Vec{1}((Dual(v[1], one(T)),))
    return v_dual
end

@inline function _load(v::Vec{2, T}) where {T}
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{2}((Dual(v[1], o, z),
                               Dual(v[2], z, o)))
    return v_dual
end

@inline function _load(v::Vec{3, T}) where {T}
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Vec{3}((Dual(v[1], o, z, z),
                               Dual(v[2], z, o, z),
                               Dual(v[3], z, z, o)))
    return v_dual
end

# Second order tensors
@inline function _load(v::Tensor{2, 1, T}) where {T}
    @inbounds v_dual = Tensor{2, 1}((Dual(get_data(v)[1], one(T)),))
    return v_dual
end

@inline function _load(v::SymmetricTensor{2, 1, T}) where {T}
    @inbounds v_dual = SymmetricTensor{2, 1}((Dual(get_data(v)[1], one(T)),))
    return v_dual
end

@inline function _load(v::Tensor{2, 2, T}) where {T}
    data = get_data(v)
    o = one(T)
    z = zero(T)
    @inbounds v_dual = Tensor{2, 2}((Dual(data[1], o, z, z, z),
                                     Dual(data[2], z, o, z, z),
                                     Dual(data[3], z, z, o, z),
                                     Dual(data[4], z, z, z, o)))
    return v_dual
end

@inline function _load(v::SymmetricTensor{2, 2, T}) where {T}
    data = get_data(v)
    o = one(T)
    o2 = T(1/2)
    z = zero(T)
    @inbounds v_dual = SymmetricTensor{2, 2}((Dual(data[1], o, z, z),
                                              Dual(data[2], z, o2, z),
                                              Dual(data[3], z, z, o)))
    return v_dual
end

@inline function _load(v::Tensor{2, 3, T}) where {T}
    data = get_data(v)
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

@inline function _load(v::SymmetricTensor{2, 3, T}) where {T}
    data = get_data(v)
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
    gradient(f::Function, v::Union{SecondOrderTensor, Vec})
    gradient(f::Function, v::Union{SecondOrderTensor, Vec}, :all)

Computes the gradient of the input function. If the (pseudo)-keyword `all`
is given, the value of the function is also returned as a second output argument.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇f = gradient(norm, A)
2×2 Tensors.SymmetricTensor{2,2,Float64,3}:
 0.434906  0.56442
 0.56442   0.416793

julia> ∇f, f = gradient(norm, A, :all);
```
"""
function gradient(f::F, v::Union{SecondOrderTensor, Vec}) where {F}
    v_dual = _load(v)
    res = f(v_dual)
    return _extract_gradient(res, v)
end
function gradient(f::F, v::Union{SecondOrderTensor, Vec}, ::Symbol) where {F}
    v_dual = _load(v)
    res = f(v_dual)
    return _extract_gradient(res, v), _extract_value(res, v)
end
const ∇ = gradient

"""
    hessian(f::Function, v::Union{SecondOrderTensor, Vec})
    hessian(f::Function, v::Union{SecondOrderTensor, Vec}, :all)

Computes the hessian of the input function. If the (pseudo)-keyword `all`
is given, the lower order results (gradient and value) of the function is
also returned as a second and third output argument.

# Examples
```jldoctest
julia> A = rand(SymmetricTensor{2, 2});

julia> ∇∇f = hessian(norm, A)
2×2×2×2 Tensors.SymmetricTensor{4,2,Float64,9}:
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
function hessian(f::F, v::Union{SecondOrderTensor, Vec}) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v)
end

function hessian(f::F, v::Union{SecondOrderTensor, Vec}, ::Symbol) where {F}
    gradf = y -> gradient(f, y)
    return gradient(gradf, v), gradient(f, v, :all)...
end
const ∇∇ = hessian

"""
    div(f, x)

Calculate the divergence of the vector field `f`, in the point `x`.

# Examples
```jldoctest
julia> f(x) = 2x;

julia> x = rand(Vec{3});

julia> div(f, x)
6.0
```
"""
Base.div(f::F, v::Vec) where {F<:Function} = tr(gradient(f, v))

"""
    curl(f, x)

Calculate the curl of the vector field `f`, in the point `x`.

# Examples
```jldoctest
julia> f(x) = Vec{3}((x[2], x[3], -x[1]));

julia> x = rand(Vec{3});

julia> curl(f, x)
3-element Tensors.Tensor{1,3,Float64,3}:
 -1.0
  1.0
 -1.0
```
"""
function curl(f::F, v::Vec{3}) where F
    @inbounds begin
        ∇f = gradient(f, v)
        c = Vec{3}((∇f[3,2] - ∇f[2,3], ∇f[1,3] - ∇f[3,1], ∇f[2,1] - ∇f[1,2]))
    end
    return c
end
curl(f::F, v::Vec{1, T}) where {F, T} = curl(f, Vec{3}((v[1], T(0), T(0))))
curl(f::F, v::Vec{2, T}) where {F, T} = curl(f, Vec{3}((v[1], v[2], T(0))))

"""
    laplace(f, x)

Calculate the laplacian of the field `f`, in the point `x`.
If `f` is a vector field, use broadcasting.

# Examples
```jldoctest
julia> x = rand(Vec{3});

julia> f(x) = norm(x);

julia> laplace(f, x)
1.7833701103136868

julia> g(x) = x*norm(x);

julia> laplace.(g, x)
3-element Tensors.Tensor{1,3,Float64,3}:
 2.10739
 2.73497
 2.01962
```
"""
function laplace(f::F, v) where F
    return div(x -> gradient(f, x), v)
end
const Δ = laplace

function Base.broadcast(::typeof(laplace), f::F, v::Vec{3}) where {F}
    @inbounds begin
        vdd = _load(_load(v))
        res = f(vdd)
        v1 = res[1].partials[1].partials[1] + res[1].partials[2].partials[2] + res[1].partials[3].partials[3]
        v2 = res[2].partials[1].partials[1] + res[2].partials[2].partials[2] + res[2].partials[3].partials[3]
        v3 = res[3].partials[1].partials[1] + res[3].partials[2].partials[2] + res[3].partials[3].partials[3]
    end
    return Vec{3}((v1, v2, v3))
end
