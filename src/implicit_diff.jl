function minimise_logsumexp(θ; T, min_decision, max_decision, initial_guess, solver=() -> ECOS.Optimizer())
    A = θ[:, 1:end-1]
    B = θ[:, end]
    m = size(A)[2]
    u = Convex.Variable(m)
    if initial_guess != nothing
        u.value = initial_guess
    end
    obj = T * Convex.logsumexp((1/T)*(A*u + B))
    prob = Convex.minimize(obj)
    if min_decision != nothing
        prob.constraints += [u >= min_decision]
    end
    if max_decision != nothing
        prob.constraints += [u <= max_decision]
    end
    solve!(prob, solver(), silent_solver=true, verbose=false)
    minimiser = typeof(u.value) <: Number ? [u.value] : u.value[:]  # to make it a vector
    return minimiser
end


function forward_lse_optim(θ; kwargs...)
    u = minimise_logsumexp(θ; kwargs...)
    z = 0
    return u, z
end


function proj_hypercube(u; min_decision, max_decision)
    if max_decision != nothing
        u = min.(max_decision, u)
    end
    if max_decision != nothing
        u = max.(min_decision, u)
    end
    return u
end

function conditions_lse_optim(θ, u, z; kwargs...)
    A = θ[:, 1:end-1]
    B = θ[:, end]
    ∇₂f = A' * Flux.softmax(A*u+B)
    η = 0.1
    return u .- proj_hypercube(u .- η .* ∇₂f; kwargs...)
end


"""
See https://github.com/gdalle/ImplicitDifferentiation.jl for details.
"""
function implicit_lse_optim(θ; T, min_decision, max_decision, initial_guess, solver)
    tmp = ImplicitFunction(
                           θ -> forward_lse_optim(θ; T, min_decision, max_decision, initial_guess, solver),
                           (θ, u, z) -> conditions_lse_optim(θ, u, z; min_decision, max_decision),
                          )
    tmp(θ)[1]
end
