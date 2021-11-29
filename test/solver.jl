using Convex
using Mosek, MosekTools
using Optim

using Transducers
using BenchmarkTools


function f(x, u)
    0.5 * (-x'*x + sum(dot(^)(u,2)))
end

function convex_solver(x, u_guess, ulim)
    m = length(u_guess)
    u = Convex.Variable(m)
    problem = minimize(f(x, u))
    problem.constraints += [u >= ulim[1]]
    problem.constraints += [u <= ulim[2]]
    solve!(problem, Mosek.Optimizer(); silent_solver=true)
    u.value, problem.optval
end

function ipnewton_solver(x, u_guess, ulim)
    dfc = TwiceDifferentiableConstraints(ulim[1], ulim[2])
    obj(u) = f(x, u)
    res = Optim.optimize(obj, dfc, u_guess, IPNewton())
    res.minimizer, res.minimum
end

@testset "solver" begin
    for N in [1, 10, 100]
        @show n, m = N, N
        d = 100
        xs = 1:d |> Map(i -> rand(n)) |> collect
        us = 1:d |> Map(i -> rand(m)) |> collect
        ulim = (-2*ones(m), 2*ones(m))
        # case 1: convex solver
        println("convex solver")
        @btime convex_solver(rand($n), rand($m), $ulim)
        println("non-convex solver (ipnewton)")
        @btime ipnewton_solver(rand($n), rand($m), $ulim)
    end
    nothing
end
