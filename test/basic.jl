using Test
using Flux
using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using UnPack
using Transducers
using Convex
using BenchmarkTools
using Mosek, MosekTools


function infer_test(approximator)
    @unpack n, m = approximator
    # single data inference
    x = rand(n)
    u = rand(m)
    @test approximator(x, u) |> size  == (1,)
    # multiple data inference
    d = 10
    _xs = rand(n, d)
    _us = rand(m, d)
    @test approximator(_xs, _us) |> size  == (1, d)
end

function optimise_test(approximator)
    @unpack n, m = approximator
    d = 100
    if typeof(approximator) <: ParametrisedConvexApproximator
        x = rand(n)
        u = Convex.Variable(m)
        @test approximator(x, u) |> size == (1,)  # inference with Convex.jl
    end
    _xs = rand(n, d)
    println("Optimise a single point")
    @time res = optimise(approximator, rand(n))
    # TODO: change to BenchmarkTools...?
    # println("Optimise a single point (analysing the result using BenchmarkTools...)")
    # @btime res = optimise($approximator, rand($n))
    println("Optimise $(d) points (using parallel computing)")
    @time res = optimise(approximator, _xs)
    @test res.minimiser |> size == (m, d)  # optimise; minimiser
    @test res.optval |> size == (1, d)  # optimise; optval
end

function training_test(approximator)
    error("TODO")
end

function test_all(approximator)
    @show typeof(approximator)
    infer_test(approximator)
    optimise_test(approximator)
    # training_test(approximator)
end

@testset "basic" begin
    # ns = [1, 10, 100]
    # ms = [1, 10, 100]
    println("TODO: change ns and ms")
    ns = [1]
    ms = [1]
    for (n, m) in zip(ns, ms)
        println("n = $(n), m = $(m)")
        i_max = 20
        T = 1e-1
        h_array = [64, 64, 64]
        act = Flux.leakyrelu
        α_is = 1:i_max |> Map(i -> Flux.glorot_uniform(n+m)) |> collect
        β_is = 1:i_max |> Map(i -> Flux.glorot_uniform(1)) |> collect
        # generate approximators
        fnn = FNN(n, m, h_array, act)
        ma = MA(α_is, β_is; n=n, m=m)
        lse = LSE(α_is, β_is, T; n=n, m=m)
        pma = PMA(n, m, i_max, h_array, act)
        plse = PLSE(n, m, i_max, T, h_array, act)
        approximators = (;
                         fnn=fnn,
                         ma=ma,
                         lse=lse,
                         pma=pma,
                         plse=plse,
                        )
        approximators |> Map(test_all) |> collect
    end
end
