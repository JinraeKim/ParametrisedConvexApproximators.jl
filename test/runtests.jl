using Test
using Random


@testset "ParametrisedConvexApproximators" begin
    include("networks.jl")
    include("dataset_trainer.jl")
end
