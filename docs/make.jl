using Documenter, Tensors

makedocs(
    modules = [Tensors],
    format = :html,
    sitename = "Tensors.jl",
    doctest = true,
    strict = true,
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
        "Demos" => "demos.md"
    ]
)

deploydocs(
    repo = "github.com/KristofferC/Tensors.jl.git",
    target = "build",
    julia = "0.5",
    deps = nothing,
    make = nothing
)
