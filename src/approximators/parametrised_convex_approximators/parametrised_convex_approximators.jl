abstract type ParametrisedConvexApproximator <: AbstractApproximator end

function affine_map(nn::ParametrisedConvexApproximator, x::Array, u::Array)
    @unpack NN, i_max, m = nn
    d = size(x)[2]
    X = reshape(NN(x), i_max, m+1, d)  # size(X1) = (i_max, m, d)
    tmp = hcat([(X[:, 1:nn.m, i]*u[:, i] .+ X[:, end:end, i]) for i in 1:d]...)
end

function affine_map(nn::ParametrisedConvexApproximator, x::Array, u::Convex.AbstractExpr)
    @unpack NN, i_max, m = nn
    X = reshape(NN(x), i_max, m+1)  # size(X1) = (i_max, m)
    tmp = (
           X[:, 1:end-1] * u + (X[:, 1:end-1]*zeros(size(u)) .+ X[:, end:end])
          )  # X1*zeros(size(u)) is for compatibility with Convex.jl
end


include("PMA.jl")
include("PLSE.jl")