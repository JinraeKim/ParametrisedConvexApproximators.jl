"""
    _optimize(approximator::ParameterizedConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `approximator::ParameterizedConvexApproximator` for given
data point `x::AbstractVector`.
Solve with Convex.jl [1].
Available solvers include SCS [2], COSMO [3], Mosek [4], etc.
# Refs.
[1] https://github.com/jump-dev/Convex.jl
[2] https://github.com/jump-dev/SCS.jl
[3] https://github.com/oxfordcontrol/COSMO.jl
[4] https://github.com/MOSEK/Mosek.jl
"""
function _optimize(approximator::ParameterizedConvexApproximator, x::AbstractVector, u_min, u_max;
        solver=SCS,
    )
    (; m) = approximator
    u = Convex.Variable(m)
    if u_min != nothing
        @assert length(u) == length(u_min)
    end
    if u_max != nothing
        @assert length(u) == length(u_max)
    end
    problem = minimize(approximator(x, u)[1])
    if u_min != nothing
        problem.constraints += [u >= u_min]
    end
    if u_max != nothing
        problem.constraints += [u <= u_max]
    end
    solve!(problem, solver.Optimizer(); verbose=false, silent_solver=true)
    minimizer = typeof(u.value) <: Number ? [u.value] : u.value[:]  # to make it a vector
    optval = [problem.optval]  # to make it a vector
    minimizer, optval
end

"""
    _optimize(approximator::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `approximator::AbstractApproximator` (non-parameterized-convex) for given
data point `x::AbstractVector`.
Default solver is `IPNewton` in Optim.jl for box constraints [1].
# Refs.
[1] https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#box-minimzation
"""
function _optimize(approximator::AbstractApproximator, x::AbstractVector, u_min, u_max)
    (; m) = approximator
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
    minimizer = prod(size(u_guess)) == 1 ? res.minimizer[1] : res.minimizer
    optval = [res.minimum]  # to make it a vector
    minimizer, optval
end

function optimize(approximator::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )
    minimizer, optval = _optimize(approximator, x, u_min, u_max)
    if minimizer == nothing
        (; m) = approximator
        minimizer = repeat([nothing], m)
    end
    result = (; minimizer=minimizer, optval=optval)
end

"""
    optimize(approximator::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `approximator::AbstractApproximator` for given
data point `x::AbstractMatrix` using multi-thread computing (powered by Transducers.jl).
"""
function optimize(approximator::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
        multithreading=true,
    )
    collector = multithreading ? Transducers.tcollect : collect
    d = size(x)[2]
    ress = 1:d |> Map(i -> optimize(approximator, x[:, i]; u_min=u_min, u_max=u_max)) |> collector
    minimizer_matrix = hcat((ress |> Map(res -> res.minimizer) |> collector)...)
    optval_matrix = hcat((ress |> Map(res -> res.optval) |> collector)...)
    (; minimizer = minimizer_matrix, optval = optval_matrix)
end
