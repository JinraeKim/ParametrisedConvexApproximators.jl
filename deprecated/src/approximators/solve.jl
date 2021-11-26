"""
Minimiser inference and optval (min-value) calculation for NormalisedApproximator.
Case 1) typeof(normalised_approximator.approximator) <: ParametrisedConvexApproximator
Convex.jl, a Disciplined Convex Programming (DCP) tool, is utilised [1].
Case 2) otherwise
Algorithm `IPNewton` in Optim.jl has been implemented here for box constraints [2].
# NOTICE
Extracting approximator::ParametrisedConvexApproximator
from normalised_approximator::NormalisedApproximator
would be a wrong result due to the absence of normalisation.
# References
[1] https://github.com/jump-dev/Convex.jl
[2] https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#box-minimzation
"""
function Convex.solve!(normalised_approximator::NormalisedApproximator, x::AbstractVector; lim=(nothing, nothing))
    approx_type = typeof(normalised_approximator.approximator)
    @unpack m = normalised_approximator.approximator
    u_min = lim[1] == nothing ? Float64[] : lim[1]  # passing empty arrays for no constraint
    u_max = lim[2] == nothing ? Float64[] : lim[2]  # passing empty arrays for no constraint
    u_guess = randn(m)
    if u_min != Float64[]
        u_guess = maximum(hcat(u_min, u_guess); dims=2)[:] + 2*eps()*ones(m)  # make it an interior point
    end
    if u_max != Float64[]
        u_guess = minimum(hcat(u_max, u_guess); dims=2)[:] - eps()*ones(m)  # make it an interior point
    end
    if approx_type == ParametrisedConvexApproximator
        u = Convex.Variable(m)
        u.value = u_guess
        problem = minimize(normalised_approximator(x, u)[1])
        if u_min != Float64[]
            problem.constraints += [u >= u_min]
        end
        if u_max != Float64[]
            problem.constraints += [u <= u_max]
        end
        solve!(problem, Mosek.Optimizer(); silent_solver=true)
        result = (; minimiser = deepcopy(u.value), optval = deepcopy(problem.optval))
    else approx_type == ParametrisedConvexApproximator
        obj(u) = normalised_approximator(x, u)[1]
        # u_min = lim[1] == nothing ? Float64[] : lim[1]  # passing empty arrays for no constraint
        # u_max = lim[2] == nothing ? Float64[] : lim[2]  # passing empty arrays for no constraint
        # u_guess = randn(m)
        # if u_min != Float64[]
        #     u_guess = maximum(hcat(u_min, u_guess); dims=2)[:] + 2*eps()*ones(m)  # make it an interior point
        # end
        # if u_max != Float64[]
        #     u_guess = minimum(hcat(u_max, u_guess); dims=2)[:] - eps()*ones(m)  # make it an interior point
        # end
        dfc = TwiceDifferentiableConstraints(u_min, u_max)
        res = Optim.optimize(obj, dfc, u_guess, IPNewton())
        minimiser = prod(size(u_guess)) == 1 ? deepcopy(res.minimizer[1]) : deepcopy(res.minimizer)
        optval = deepcopy(res.minimum)
        result = (; minimiser=minimiser, optval=optval)  # NamedTuple
    end
    return result
end

function Convex.solve!(normalised_approximator::NormalisedApproximator, x::AbstractMatrix; lim=(nothing, nothing))
    d = size(x)[2]
    ress = 1:d |> Map(i -> solve!(normalised_approximator, x[:, i]; lim=lim)) |> tcollect
    minimiser_matrix = hcat((ress |> Map(res -> res.minimiser) |> collect)...)
    optval_matrix = hcat((ress |> Map(res -> res.optval) |> collect)...)
    (; minimiser = minimiser_matrix, optval = optval_matrix)
end
