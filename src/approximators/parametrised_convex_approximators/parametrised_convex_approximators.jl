abstract type ParametrisedConvexApproximator <: AbstractApproximator end

function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::AbstractArray)
    @unpack NN, i_max, m = nn
    d = size(x)[2]
    X = reshape(NN(x), i_max, m+1, d)  # size(X1) = (i_max, m, d)
    tmp = hcat([(X[:, 1:nn.m, i]*u[:, i] .+ X[:, end:end, i]) for i in 1:d]...)
end

function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::Convex.AbstractExpr)
    @unpack NN, i_max, m = nn
    X = reshape(NN(x), i_max, m+1)  # size(X1) = (i_max, m)
    tmp = (
           X[:, 1:end-1] * u + (X[:, 1:end-1]*zeros(size(u)) .+ X[:, end:end])
          )  # X1*zeros(size(u)) is for compatibility with Convex.jl
end

"""
# NOTICE
Extracting approximator::ParametrisedConvexApproximator
from normalised_approximator::NormalisedApproximator
would be a wrong result due to the absence of normalisation.
"""
function Convex.solve!(normalised_approximator::NormalisedApproximator, x::AbstractVector)
    typeof(normalised_approximator.approximator) <: ParametrisedConvexApproximator ? nothing : error("Convex.solve! can be used only for typeof(normalised_approximator.approximator) <: ParametrisedConvexApproximator")
    @unpack m = normalised_approximator.approximator
    u = Convex.Variable(m)
    problem = minimize(normalised_approximator(x, u)[1])
    solve!(problem, Mosek.Optimizer(); silent_solver=true)
    (; minimiser = u.value, optval = problem.optval)
end

function Convex.solve!(normalised_approximator::NormalisedApproximator, x::AbstractMatrix)
    typeof(normalised_approximator.approximator) <: ParametrisedConvexApproximator ? nothing : error("Convex.solve! can be used only for typeof(normalised_approximator.approximator) <: ParametrisedConvexApproximator")
    d = size(x)[2]
    ress = 1:d |> Map(i -> solve!(normalised_approximator, x[:, i])) |> tcollect
    minimiser_matrix = hcat((ress |> Map(res -> res.minimiser) |> collect)...)
    optval_matrix = hcat((ress |> Map(res -> res.optval) |> collect)...)
    (; minimiser = minimiser_matrix, optval = optval_matrix)
end

include("PMA.jl")
include("PLSE.jl")
