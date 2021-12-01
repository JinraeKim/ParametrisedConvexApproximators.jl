using Test
using Flux
using Flux: DataLoader
using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using UnPack
using Transducers
using Convex
using BenchmarkTools
using Random
using Plots


function target_function(x, u)
    _g = zeros(size(x)...)
    for (i, _x) in enumerate(x)
        if _x > 0
            _g[i] = (_x)^(1/4)
        else
            _g[i] = (_x)^(4)
        end
    end
    [sum(_g) + 0.5*u'*u]
end

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

function optimise_test(approximator, data)
    @unpack n, m = approximator
    d = data.x |> length
    println("optimise test; with $(d) test data")
    if typeof(approximator) <: ParametrisedConvexApproximator
        x = rand(n)
        u = Convex.Variable(m)
        @test approximator(x, u) |> size == (1,)  # inference with Convex.jl
    end
    _xs = hcat(data.x...)
    println("Optimise a single point")
    @time _ = optimise(approximator, rand(n))
    # TODO: change to BenchmarkTools...?
    # println("Optimise a single point (analysing the result using BenchmarkTools...)")
    # @btime res = optimise($approximator, rand($n))
    println("Optimise $(d) points (using parallel computing)")
    @time res = optimise(approximator, _xs)
    # @time res = optimise(approximator, _xs; u_min=u_min, u_max=u_max)  # with box constraints
    @test res.minimiser |> size == (m, d)  # optimise; minimiser
    @test res.optval |> size == (1, d)  # optimise; optval
    # compare true and estimated minimisers and optvals
    @warn("If you change the target function, you may have to change the true minimisers manually.")
    minimisers_true = 1:d |> Map(i -> zeros(m)) |> collect
    optvals_true = 1:d |> Map(i -> target_function(data.x[i], minimisers_true[i])) |> collect
    minimisers_estimated = 1:d |> Map(i -> res.minimiser[:, i]) |> collect
    optvals_estimated = 1:d |> Map(i -> res.optval[:, i]) |> collect
    minimisers_diff_norm = 1:d |> Map(i -> norm(minimisers_estimated[i] - minimisers_true[i])) |> collect
    optvals_diff_norm = 1:d |> Map(i -> norm(optvals_estimated[i] - optvals_true[i])) |> collect
    println("norm(estimated minimiser - true minimiser)'s mean: $(mean(minimisers_diff_norm))")
    println("norm(estimated optval - true minimiser)'s mean: $(mean(optvals_diff_norm))")
end

function training_test(approximator, data_train, data_test, epochs)
    loss(d) = Flux.Losses.mse(approximator(d.x, d.u), d.f)
    opt = ADAM(1e-3)
    ps = Flux.params(approximator)
    dataloader = DataLoader(data_train; batchsize=16, shuffle=true, partial=false)
    println("Training $(epochs) epoch...")
    for epoch in 0:epochs
        println("epoch: $(epoch) / $(epochs)")
        if epoch != 0
            for d in dataloader
                train_loss, back = Flux.Zygote.pullback(() -> loss(d), ps)
                gs = back(one(train_loss))
                Flux.update!(opt, ps, gs)
            end
        end
        @show loss(data_test)
    end
end

function test_all(approximator, data, epochs)
    # split data
    xs_us_fs = zip(data.x, data.u, data.f) |> collect
    xs_us_fs_train, xs_us_fs_test = partitionTrainTest(xs_us_fs, 0.8)  # 80:20
    data_train = (;
                  x=hcat((xs_us_fs_train |> Map(xuf -> xuf[1]) |> collect)...),
                  u=hcat((xs_us_fs_train |> Map(xuf -> xuf[2]) |> collect)...),
                  f=hcat((xs_us_fs_train |> Map(xuf -> xuf[3]) |> collect)...),
                 )
    data_test = (;
                 x=hcat((xs_us_fs_test |> Map(xuf -> xuf[1]) |> collect)...),
                 u=hcat((xs_us_fs_test |> Map(xuf -> xuf[2]) |> collect)...),
                 f=hcat((xs_us_fs_test |> Map(xuf -> xuf[3]) |> collect)...),
                )
    @show typeof(approximator)
    infer_test(approximator)
    training_test(approximator, data_train, data_test, epochs)
    optimise_test(approximator, data_test)
end

function plot_figures(approximator, min, max)
    fig = plot()
    error("TODO")
end

@testset "basic" begin
    sample(min, max) = min + (max - min) .* rand(size(min)...)
    # tests
    # ns = [1, 10, 100]
    # ms = [1, 10, 100]
    println("TODO: change ns and ms, epochs, etc.")
    ns = [1]
    ms = [1]
    epochs = 1
    for (n, m) in zip(ns, ms)
        Random.seed!(2021)
        # training data
        d = 1_000
        min = (; x = -1*ones(n), u = -1*ones(m))
        max = (; x = 1*ones(n), u = 1*ones(m))
        xs = 1:d |> Map(i -> sample(min.x, max.x)) |> collect
        us = 1:d |> Map(i -> sample(min.u, max.u)) |> collect
        fs = zip(xs, us) |> MapSplat((x, u) -> target_function(x, u)) |> collect
        data = (; x=xs, u=us, f=fs)
        println("n = $(n), m = $(m)")
        i_max = 20
        T = 1e-1
        h_array = [512, 512]
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
        approximators |> Map(approximator -> test_all(approximator, data, epochs)) |> collect
        if n == 1 && m == 1
            approximators |> Map(approximator -> plot_figures(approximator, min, max)) |> collect
        end
    end
end
