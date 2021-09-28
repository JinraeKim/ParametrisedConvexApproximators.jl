using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using Test

using Flux
using Transducers
using Convex
using Random
using ForwardDiff


"""
test function
"""
function f(x, u)
    0.5 * (-x'*x + u'*u)
end

function supervised_learning!(approximator, xuf_data)
    @show approximator |> typeof
    xuf_data_train, xuf_data_test = PCA.partitionTrainTest(xuf_data)
    PCA.train_approximator!(approximator, xuf_data_train, xuf_data_test;
                            epochs=10,
                           )
    @show "No error while training the approximator"
end

function test(approximator, _xs, _us)
    @show approximator |> typeof
    d = size(_xs)[2]
    m = size(_us)[1]
    _x = _xs[:, 1]
    _u = _us[:, 1]
    u_convex = Convex.Variable(length(_us[:, 1]))
    u_convex.value = _u
    @testset "infer size check" begin
        @test approximator(_xs, _us) |> size == (1, d)
        @test approximator(_x, _u) |> size == (1,)
    end
    @testset "Convex.Variable evaluation check" begin
        @test approximator(_x, u_convex.value) ≈ approximator(_x, _u)
        @test evaluate.(approximator(_x, u_convex)) ≈ approximator(_x, _u)
    end
end

@testset "Apporixmators" begin
    seed = 2021
    Random.seed!(seed)
    # default
    n, m, d = 2, 1, 100
    i_max = 10
    h_array = [2, 2]
    T = 1e-1
    act = Flux.leakyrelu
    xs = 1:d |> Map(i -> 5*(2*(rand(n) .- 0.5))) |> collect
    us = 1:d |> Map(i -> 5*(2*(rand(m) .- 0.5))) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    xuf_data = PCA.xufData(xs, us, fs)
    _xs = hcat(xs...)
    _us = hcat(us...)
    u_is = range(-1, 1, length=i_max) |> Map(_u_i -> [_u_i]) |> collect  # to make it a matrix
    # u_star_is = u_is |> Map(u_i -> (x -> f_partial_u(x, u_i))) |> collect
    u_star_is = u_is |> Map(u_i -> (x -> ForwardDiff.gradient(u -> f(x, u), u_i))) |> collect
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
    # push!(approximators, pma_theoretical)  # TODO: use ForwardDiff to obtain partial derivative?
    push!(approximators, plse)
    # test
    print("Testing basic functionality...")
    approximators |> Map(approx -> test(approx, _xs, _us)) |> collect
    # training
    print("Testing supervised_learning...")
    approximators |> Map(approx -> supervised_learning!(approx, xuf_data)) |> collect
    nothing
end
