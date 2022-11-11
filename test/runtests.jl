using Test
using Random


@testset "ParameterizedConvexApproximators" begin
    include("networks.jl")
    include("dataset_trainer.jl")
end
