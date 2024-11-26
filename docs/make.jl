using Documenter, Tensors

# Setup for doctests in docstrings
DocMeta.setdocmeta!(Tensors, :DocTestSetup,
    quote
        using Random
        Random.seed!(1234)
        using Tensors
    end
)

makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Tensors],
    warnonly = true,
    sitename = "Tensors.jl",
    pages = Any[
        "Home" => "index.md",
        "Manual" => [
            "man/constructing_tensors.md",
            "man/indexing.md",
            "man/binary_operators.md",
            "man/other_operators.md",
            "man/storing_tensors.md",
            "man/automatic_differentiation.md",
            ],
        "Benchmarks" => "benchmarks.md",
        "Demos" => "demos.md"
    ]
)

deploydocs(
    repo = "github.com/Ferrite-FEM/Tensors.jl.git",
)
