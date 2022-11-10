using Test
using Random


@testset "ParameterizedConvexApproximators" begin
    include("networks.jl")
    include("data_manipulation.jl")
    include("dataset_trainer.jl")
end
