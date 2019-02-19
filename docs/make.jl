using Documenter, Tensors

makedocs(
    format = Documenter.HTML(prettyurls = get(ENV, "CI", nothing) == "true"),
    modules = [Tensors],
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
    repo = "github.com/KristofferC/Tensors.jl.git",
)
