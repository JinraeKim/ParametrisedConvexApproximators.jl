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
    _optimize(approximator::DifferenceOfConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `approximator::DifferenceOfConvexApproximator` for given data point `x::AbstractVector`.

Basic DCA [1] is used.
# Refs.
[1] H. A. Le Thi and T. Pham Dinh, “DC programming and DCA: thirty years of developments,” Math. Program., vol. 169, no. 1, pp. 5–68, May 2018, doi: 10.1007/s10107-018-1235-y.
[2] https://github.com/Corrado-possieri/DLSE_neural_networks/commit/8883e5bcf1733b79b2dd3c432b31af30b4bba0a6#diff-aa888e053028cc6dbd9f0cfb1c30f61f1bde256be213f27b9a083b95292ec5ebR26
"""
function _optimize(approximator::DifferenceOfConvexApproximator, x::AbstractVector, u_min, u_max;
        solver=SCS,
        max_iter=30,
        tol=1e-3,  # borrowed from [2]
    )
    (; m) = approximator.NN1
    u = Convex.Variable(m)
    if u_min != nothing
        @assert length(u) == length(u_min)
    end
    if u_max != nothing
        @assert length(u) == length(u_max)
    end
    # initial guess
    χ = u_min + (u_max - u_min) .* rand(size(u_min)...)
    grad_NN2(u) = ForwardDiff.gradient(u -> approximator.NN2(x, u)[1], u)
    optval = Inf
    χ_next = nothing
    k = 0
    # @time while true
    while true
        k = k + 1
        v = grad_NN2(χ)  # BE CAREFUL: CONSIDER THAT IT IS FOR BIVARIATE FUNCTION
        problem = minimize(approximator.NN1(x, u)[1] - v'*u)
        if u_min != nothing
            problem.constraints += [u >= u_min]
        end
        if u_max != nothing
            problem.constraints += [u <= u_max]
        end
        solve!(problem, solver.Optimizer(); verbose=false, silent_solver=true)
        χ_next = typeof(u.value) <: Number ? [u.value] : u.value[:]  # to make it a vector
        optval = [problem.optval]
        if norm(χ_next - χ) / (1+norm(χ)) < tol || k == max_iter
            # @show k
            # @show χ, χ_next
            break
        end
        χ = χ_next
    end
    minimizer = χ_next
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
