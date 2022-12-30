using SnoopPrecompile

@precompile_all_calls begin
    for dim in (2, 3)
        v = ones(Tensor{1, dim, Float64,})
        σ = one(SymmetricTensor{2, dim, Float64})
        F = one(Tensor{2, dim, Float64,})
        E = one(SymmetricTensor{4, dim, Float64})
        C = one(Tensor{4, dim, Float64,})
        v * 1.0
        v ⋅ v
        v ⊗ v
        σ ⋅ v
        F ⋅ v
        σ ⊗ σ
        F ⊗ F
        σ * 1.0
        F * 1.0
        σ ⊡ σ
        F ⊡ F
        E ⊡ σ
        C ⊡ F
        E * 1.0
        C * 1.0

        # TODO: AD?
    end

    # See discussion in https://github.com/Ferrite-FEM/Tensors.jl/pull/190
    if @isdefined var"#102#103"
        precompile(Tuple{typeof(apply_all), Type{Tensor{1, 2, T, M} where M where T}, var"#102#103"{Float64}})
        precompile(Tuple{typeof(apply_all), Type{Tensor{1, 3, T, M} where M where T}, var"#102#103"{Float64}})
    end
end
