using Test
using Random


@testset "ParametrisedConvexApproximators" begin
    include("networks.jl")
    include("pure_train.jl")
    include("dataset_trainer.jl")
end
