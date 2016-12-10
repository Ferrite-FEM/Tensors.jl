using Documenter, ContMechTensors

makedocs(
    modules = [ContMechTensors],
    format = :html,
    sitename = "ContMechTensors.jl",
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
    repo = "github.com/KristofferC/ContMechTensors.jl.git",
    target = "build",
    deps = nothing,
    make = nothing,
)