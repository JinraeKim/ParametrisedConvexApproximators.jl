using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using Test

using Flux
using Transducers
using Convex


"""
test function
"""
function f(x, u)
    0.5 * (-x'*x + u'*u)
end

function f_partial_u(x, u)
    u
end

function test(approximator, _xs, _us)
    @show approximator |> typeof
    @show Flux.params(approximator)
    d = size(_xs)[2]
    m = size(_us)[1]
    u_convex = Convex.Variable(length(_us[:, 1]))
    u_convex.value = rand(m)
    _x = _xs[:, 1]
    _u = _us[:, 1]
    @testset "infer_test" begin
        approximator(_xs, _us) |> size == (1, d)
        approximator(_x, _u) |> size == (1,)
        evaluate(approximator(_x, u_convex)) == approximator(_x, _u)
    end
end

function main()
    # default
    n, m, d = 2, 1, 10
    i_max = 5
    h_array = [1, 1]
    T = 1e-1
    act = Flux.relu
    _xs = rand(n, d)
    _us = rand(m, d)
    u_is = range(-1, 1, length=i_max) |> Map(_u_i -> [_u_i]) |> collect  # to make it a matrix
    u_star_is = u_is |> Map(u_i -> (x -> f_partial_u(x, u_i))) |> collect
    α_is = 1:i_max |> Map(i -> rand(n+m)) |> collect
    β_is = 1:i_max |> Map(i -> rand(1)) |> collect
    # test
    ma = MA(α_is, β_is)
    lse = LSE(α_is, β_is, T)
    pma_basic = PMA(n, m, i_max, h_array, act)
    pma_theoretical = PMA(n, m, u_is, u_star_is, f)
    plse = PLSE(n, m, i_max, T, h_array, act)
    approximators = []
    push!(approximators, ma)
    push!(approximators, lse)
    push!(approximators, pma_basic)
    push!(approximators, pma_theoretical)
    push!(approximators, plse)
    approximators |> Map(approx -> test(approx, _xs, _us)) |> collect
end
