using Test
using Random


@testset "ParametrisedConvexApproximators" begin
    seed = 2021
    Random.seed!(seed)
    core()
    # auxiliary()
end

function core()
    include("basic.jl")
end

function auxiliary()
    include("solver.jl")
end
