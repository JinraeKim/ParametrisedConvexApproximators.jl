"""
    _optimise(network::ParametrisedConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `network::ParametrisedConvexApproximator` for given
data point `x::AbstractVector`.
Solve with Convex.jl [1].
Available solvers include SCS [2], COSMO [3], Mosek [4], etc.
# Refs.
[1] https://github.com/jump-dev/Convex.jl
[2] https://github.com/jump-dev/SCS.jl
[3] https://github.com/oxfordcontrol/COSMO.jl
[4] https://github.com/MOSEK/Mosek.jl
"""
function _optimise(network::ParametrisedConvexApproximator, x::AbstractVector, u_min, u_max, initial_guess;
        solver=SCS,
    )
    (; m) = network
    u = Convex.Variable(m)
    # initial guess
    if initial_guess != nothing
        u.value = initial_guess
    end
    if u_min != nothing
        @assert length(u) == length(u_min)
    end
    if u_max != nothing
        @assert length(u) == length(u_max)
    end
    problem = minimize(network(x, u)[1])
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
    _optimise(network::DifferenceOfConvexApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `network::DifferenceOfConvexApproximator` for given data point `x::AbstractVector`.

Basic DCA [1] is used.
# Refs.
[1] H. A. Le Thi and T. Pham Dinh, “DC programming and DCA: thirty years of developments,” Math. Program., vol. 169, no. 1, pp. 5–68, May 2018, doi: 10.1007/s10107-018-1235-y.
[2] https://github.com/Corrado-possieri/DLSE_neural_networks/commit/8883e5bcf1733b79b2dd3c432b31af30b4bba0a6#diff-aa888e053028cc6dbd9f0cfb1c30f61f1bde256be213f27b9a083b95292ec5ebR26
"""
function _optimise(network::DifferenceOfConvexApproximator, x::AbstractVector, u_min, u_max, initial_guess;
        solver=SCS,
        max_iter=30,
        tol=1e-3,  # borrowed from [2]
    )
    (; m) = network.NN1
    u = Convex.Variable(m)
    if u_min != nothing
        @assert length(u) == length(u_min)
    end
    if u_max != nothing
        @assert length(u) == length(u_max)
    end
    # initial guess
    if initial_guess == nothing
        if u_min != nothing && u_max != nothing
            initial_guess = u_min + (u_max - u_min) .* rand(size(u_min)...)
        else
            initial_guess = randn(m)
            if u_min != nothing
                initial_guess = maximum(hcat(u_min, initial_guess); dims=2)[:]
            end
            if u_max != nothing
                initial_guess = minimum(hcat(u_max, initial_guess); dims=2)[:]
            end
        end
    end
    χ = initial_guess
    grad_NN2(u) = ForwardDiff.gradient(u -> network.NN2(x, u)[1], u)
    optval = Inf
    χ_next = nothing
    k = 0
    # @time while true
    while true
        k = k + 1
        v = grad_NN2(χ)  # BE CAREFUL: CONSIDER THAT IT IS FOR BIVARIATE FUNCTION
        problem = minimize(network.NN1(x, u)[1] - v'*u)
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
    _optimise(network::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `network::AbstractApproximator` (non-parameterized-convex) for given
data point `x::AbstractVector`.
Default solver is `IPNewton` in Optim.jl for box constraints [1].
# Refs.
[1] https://julianlsolvers.github.io/Optim.jl/stable/#examples/generated/ipnewton_basics/#box-minimzation
"""
function _optimise(network::AbstractApproximator, x::AbstractVector, u_min, u_max, initial_guess)
    (; m) = network
    obj(u) = network(x, u)[1]
    if u_min == nothing
        u_min = Float64[]  # no constraint
    end
    if u_max == nothing
        u_max = Float64[]  # no constraint
    end
    if initial_guess == nothing
        if u_min != Float64[] && u_max != Float64[]
            initial_guess = (u_min+eps()*ones(m)) + ((u_max-eps()*ones(m)) - (u_min+eps()*ones(m))) .* rand(size(u_min)...)
        else
            initial_guess = randn(m)
            if u_min != Float64[]
                initial_guess = maximum(hcat(u_min, initial_guess); dims=2)[:] + eps()*ones(m)  # make it an interior point
            end
            if u_max != Float64[]
                initial_guess = minimum(hcat(u_max, initial_guess); dims=2)[:] - eps()*ones(m)  # make it an interior point
            end
        end
    end
    dfc = TwiceDifferentiableConstraints(u_min, u_max)
    res = Optim.optimize(obj, dfc, initial_guess, IPNewton())
    minimizer = prod(size(initial_guess)) == 1 ? res.minimizer[1] : res.minimizer
    optval = [res.minimum]  # to make it a vector
    minimizer, optval
end


function _optimise(nn::NormalisedApproximator, x::AbstractVector, u_min, u_max, initial_guess)
    x = normalise(nn, x, :condition)
    u_min = u_min != nothing ? normalise(nn, u_min, :decision) : u_min
    u_max = u_max != nothing ? normalise(nn, u_max, :decision) : u_max
    initial_guess = initial_guess != nothing ? normalise(nn, initial_guess, :decision) : nothing
    minimizer, optval = _optimise(nn.network, x, u_min, u_max, initial_guess)
    minimizer = unnormalise(nn, minimizer, :decision)
    optval = unnormalise(nn, optval, :cost)
    return minimizer, optval
end


function optimise(network::AbstractApproximator, x::AbstractVector;
        u_min=nothing, u_max=nothing, initial_guess=nothing
    )
    minimizer, optval = _optimise(network, x, u_min, u_max, initial_guess)
    if minimizer == nothing
        (; m) = network
        minimizer = repeat([nothing], m)
    end
    result = (; minimizer=minimizer, optval=optval)
end

"""
    optimise(network::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
    )

Find a minimizer and optimal value (optval) of `network::AbstractApproximator` for given
data point `x::AbstractMatrix` using multi-thread computing (powered by Transducers.jl).
"""
function optimise(network::AbstractApproximator, x::AbstractMatrix;
        u_min=nothing, u_max=nothing,
        multithreading=true,
        initial_guess=nothing,
    )
    collector = multithreading ? Transducers.tcollect : collect
    d = size(x)[2]
    # initial guess
    initial_guess = 1:d |> Map(i -> initial_guess == nothing ? nothing : initial_guess[:, i]) |> collect
    # optimisation
    ress = 1:d |> Map(i -> optimise(network, x[:, i]; u_min=u_min, u_max=u_max, initial_guess=initial_guess[i])) |> collector
    minimizer_matrix = hcat((ress |> Map(res -> res.minimizer) |> collector)...)
    optval_matrix = hcat((ress |> Map(res -> res.optval) |> collector)...)
    (; minimizer = minimizer_matrix, optval = optval_matrix)
end
