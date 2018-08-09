using Documenter, Tensors

makedocs(
    modules = [Tensors],
    format = :html,
    sitename = "Tensors.jl",
    doctest = true,
    strict = VERSION.minor == 7 && sizeof(Int) == 8, # only strict mode on 0.7 and Int64
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
    target = "build",
    julia = "0.7",
    deps = nothing,
    make = nothing
)
