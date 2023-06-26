function minimise_logsumexp(θ; T, u_min, u_max, initial_guess, solver=() -> ECOS.Optimizer())
    A = θ[:, 1:end-1]
    B = θ[:, end]
    m = size(A)[2]
    u = Convex.Variable(m)
    if initial_guess != nothing
        u.value = initial_guess
    end
    obj = T * Convex.logsumexp((1/T)*(A*u + B))
    prob = Convex.minimize(obj)
    if u_min != nothing
        prob.constraints += [u >= u_min]
    end
    if u_max != nothing
        prob.constraints += [u <= u_max]
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


function proj_hypercube(u; u_min, u_max)
    if u_max != nothing
        u = min.(u_max, u)
    end
    if u_max != nothing
        u = max.(u_min, u)
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
function implicit_lse_optim(θ; T, u_min, u_max, initial_guess, solver)
    tmp = ImplicitFunction(
                           θ -> forward_lse_optim(θ; T, u_min, u_max, initial_guess, solver),
                           (θ, u, z) -> conditions_lse_optim(θ, u, z; u_min, u_max),
                          )
    tmp(θ)[1]
end