using Test
using Random


@testset "ParametrisedConvexApproximators" begin
    seed = 2021
    Random.seed!(seed)
    include("approximators.jl")
end
