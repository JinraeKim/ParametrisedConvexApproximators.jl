abstract type ParametrisedConvexApproximator <: AbstractApproximator end


function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::AbstractArray)
    @unpack NN, i_max, m = nn
    # @unpack NN1, NN2, i_max, m = nn
    d = size(x)[2]
    X = reshape(NN(x), i_max, m+1, d)
    tmp = hcat([(X[:, 1:end-1, i]*u[:, i] .+ X[:, end:end, i]) for i in 1:d]...)
    # X1 = reshape(NN1(x), i_max, m, d)  # size(X1) = (i_max, m, d)
    # X2 = reshape(NN2(x), i_max, 1, d)  # size(X1) = (i_max, m, d)
    # tmp = hcat([(X1[:, :, i]*u[:, i] .+ X2[:, :, i]) for i in 1:d]...)
end

function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::Convex.AbstractExpr)
    @unpack NN, i_max, m = nn
    # @unpack NN1, NN2, i_max, m = nn
    X = reshape(NN(x), i_max, m+1)  # size(X1) = (i_max, m)
    tmp = (
           X[:, 1:end-1] * u + (X[:, 1:end-1]*zeros(size(u)) .+ X[:, end:end])
          )  # X1*zeros(size(u)) is for compatibility with Convex.jl
    # X1 = reshape(NN1(x), i_max, m)  # size(X1) = (i_max, m)
    # X2 = reshape(NN2(x), i_max, 1)  # size(X1) = (i_max, m)
    # tmp = (
    #        X1 * u + (X1*zeros(size(u)) .+ X2)
    #       )  # X1*zeros(size(u)) is for compatibility with Convex.jl
end


include("PMA.jl")
include("PLSE.jl")
include("convex_approximators/convex_approximators.jl")
