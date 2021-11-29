"""
    optimise(approximator::ParametrisedConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimiser and optimal value (optval) of `approximator::ParametrisedConvexApproximator` for given
data point `x::AbstractVector`.
Default solver is Mosek with Convex.jl [1].
# Refs.
[1] https://github.com/jump-dev/Convex.jl
"""
function optimise(approximator::ParametrisedConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )
    @unpack m = approximator
    u = Convex.Variable(m)
    problem = minimize(approximator(x, u)[1])
    if u_min != nothing
        problem.constraints += [u >= u_min]
    end
    if u_max != nothing
        problem.constraints += [u <= u_max]
    end
    solve!(problem, Mosek.Optimizer(); silent_solver=true)
    result = (; minimiser=u.value, optval=problem.optval)
end

"""
    optimise(approximator::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimiser and optimal value (optval) of `approximator::AbstractApproximator` (non-parametrised-convex) for given
data point `x::AbstractVector`.
Default solver is `IPNewton` in Optim.jl for box constraints [1].
# Refs.
[1] https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#box-minimzation
"""
function optimise(approximator::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )
    @unpack m = approximator
    obj(u) = approximator(x, u)[1]
    if u_min == nothing
        u_min = Float64[]  # no constraint
    end
    if u_max == nothing
        u_max = Float64[]  # no constraint
    end
    # TODO: initial guess; to make sure that `u_min + eps*ones(m) <= u_guess <= u_max - eps*ones(m)`; this algorithm needs an initial guess in the interior of given box constraints.
    u_guess = randn(m)
    if u_min != Float64[]
        u_guess = maximum(hcat(u_min, u_guess); dims=2)[:] + 2*eps()*ones(m)  # make it an interior point
    end
    if u_max != Float64[]
        u_guess = minimum(hcat(u_max, u_guess); dims=2)[:] - eps()*ones(m)  # make it an interior point
    end
    dfc = TwiceDifferentiableConstraints(u_min, u_max)
    res = Optim.optimize(obj, dfc, u_guess, IPNewton())
    minimiser = prod(size(u_guess)) == 1 ? res.minimizer[1] : res.minimizer
    optval = res.minimum
    result = (; minimiser=minimiser, optval=optval)  # NamedTuple
end

"""
    optimise(approximator::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
    )

Find a minimiser and optimal value (optval) of `approximator::AbstractApproximator` for given
data point `x::AbstractMatrix` using multi-thread computing (powered by Transducers.jl).
"""
function optimise(approximator::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
        collector=Transducers.tcollect,
    )
    d = size(x)[2]
    ress = 1:d |> Map(i -> optimise(approximator, x[:, i]; u_min=u_min, u_max=u_max)) |> collector
    minimiser_matrix = hcat((ress |> Map(res -> res.minimiser) |> collector)...)
    optval_matrix = hcat((ress |> Map(res -> res.optval) |> collector)...)
    (; minimiser = minimiser_matrix, optval = optval_matrix)
end
