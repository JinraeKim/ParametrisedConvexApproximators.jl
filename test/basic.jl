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


__dir_save = "test/basic"


function target_function(x, u)
    # case 1
    # _g = zeros(size(x)...)
    # for (i, _x) in enumerate(x)
    #     if _x > 0
    #         _g[i] = (_x)^(1/4)
    #     else
    #         _g[i] = (_x)^(4)
    #     end
    # end
    # [sum(_g) + 0.5*u'*u]
    # case 2
    [-0.5*x'*x + 0.5*u'*u]
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

function optimise_test(approximator, data, dir_save)
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
    # @time res = optimise(approximator, _xs)
    @time res = optimise(approximator, _xs)  # with box constraints
    @test res.minimiser |> size == (m, d)  # optimise; minimiser
    @test res.optval |> size == (1, d)  # optimise; optval
    # compare true and estimated minimisers and optvals
    minimisers_true = 1:d |> Map(i -> zeros(m)) |> collect
    optvals_true = 1:d |> Map(i -> target_function(data.x[i], minimisers_true[i])) |> collect
    minimisers_estimated = 1:d |> Map(i -> res.minimiser[:, i]) |> collect
    optvals_estimated = 1:d |> Map(i -> res.optval[:, i]) |> collect
    failure_cases = findall(x -> x == repeat([nothing], m), minimisers_estimated)
    if length(failure_cases) > 0
        @warn("(optimisaiton failure) there is at least one failure case to solve the optimisation;
                # of failure cases: ($(length(failure_cases)) / $(d))")
    else
        minimisers_diff_norm = 1:d |> Map(i -> norm(minimisers_estimated[i] - minimisers_true[i])) |> collect
        optvals_diff = 1:d |> Map(i -> abs(optvals_estimated[i][1] - optvals_true[i][1])) |> collect
        println("norm(estimated minimiser - true minimiser)'s mean: $(mean(minimisers_diff_norm))")
        println("norm(estimated optval - true optval)'s mean: $(mean(optvals_diff))")
        fig_minimiser_diff_norm = histogram(minimisers_diff_norm)
        fig_optval_diff = histogram(optvals_diff)
        savefig(fig_minimiser_diff_norm, joinpath(dir_save, "minimiser_diff_norm.png"))
        savefig(fig_optval_diff, joinpath(dir_save, "optval_diff.png"))
    end
end

function training_test(approximator, data_train, data_test, epochs)
    _data_train = (;
                   x = hcat(data_train.x...),
                   u = hcat(data_train.u...),
                   f = hcat(data_train.f...),
                  )
    _data_test = (;
                   x = hcat(data_test.x...),
                   u = hcat(data_test.u...),
                   f = hcat(data_test.f...),
                  )
    loss(d) = Flux.Losses.mse(approximator(d.x, d.u), d.f)
    opt = ADAM(1e-3)
    ps = Flux.params(approximator)
    dataloader = DataLoader(_data_train; batchsize=16, shuffle=true, partial=false)
    println("Training $(epochs) epoch...")
    for epoch in 0:epochs
        println("epoch: $(epoch) / $(epochs)")
        if epoch != 0
            for d in dataloader
                train_loss, back = Flux.Zygote.pullback(() -> loss(d), ps)
                gs = back(one(train_loss))
                Flux.update!(opt, ps, gs)
                # projection for PICNN
                if typeof(approximator) == PICNN
                    for layer in approximator.NN.layers
                        if isdefined(layer, :Wz)
                            project_nonnegative!(layer.Wz)
                            if !all(layer.Wz .>= 0.0)
                                error("Projection seems not work")
                            end
                        end
                    end
                end
            end
        end
        @show loss(_data_test)
    end
end

function test_all(approximator, data, epochs, _dir_save)
    # split data
    xs_us_fs = zip(data.x, data.u, data.f) |> collect
    xs_us_fs_train, xs_us_fs_test = partitionTrainTest(xs_us_fs, 0.8)  # 80:20
    data_train = (;
                  x=xs_us_fs_train |> Map(xuf -> xuf[1]) |> collect,
                  u=xs_us_fs_train |> Map(xuf -> xuf[2]) |> collect,
                  f=xs_us_fs_train |> Map(xuf -> xuf[3]) |> collect,
                 )
    data_test = (;
                  x=xs_us_fs_test |> Map(xuf -> xuf[1]) |> collect,
                  u=xs_us_fs_test |> Map(xuf -> xuf[2]) |> collect,
                  f=xs_us_fs_test |> Map(xuf -> xuf[3]) |> collect,
                )
    @show typeof(approximator)
    @unpack n, m = approximator
    dir_save = nothing
    if typeof(approximator) == FNN
        dir_save = joinpath(_dir_save, "FNN")
    elseif typeof(approximator) == MA
        dir_save = joinpath(_dir_save, "MA")
    elseif typeof(approximator) == LSE
        dir_save = joinpath(_dir_save, "LSE")
    elseif typeof(approximator) == PMA
        dir_save = joinpath(_dir_save, "PMA")
    elseif typeof(approximator) == PLSE
        dir_save = joinpath(_dir_save, "PLSE")
    elseif typeof(approximator) == PICNN
        dir_save = joinpath(_dir_save, "PICNN")
    else
        error("Specify save directory")
    end
    mkpath(dir_save)
    # tests
    infer_test(approximator)
    @time training_test(approximator, data_train, data_test, epochs)
    optimise_test(approximator, data_test, dir_save)
    if n == 1 && m == 1
        plot_surface(approximator, min, max)
    end
end

function plot_surface(approximator, min, max)
    fig = plot()
    error("TODO")
end

@testset "basic" begin
    @warn("Note: if you change the target function, you may have to change the true minimisers manually (in the function `optimise_test`).")
    sample(min, max) = min + (max - min) .* rand(size(min)...)
    # tests
    println("TODO: change ns and ms, epochs_list, etc.")
    ns = [1]
    ms = [1]
    # ns = [1, 10, 100]
    # ms = [1, 10, 100]
    for (n, m) in zip(ns, ms)
        epochs_list = [10]
        # epochs_list = [10, 50]
        for epochs in epochs_list
            Random.seed!(2021)
            # training data
            d = 1_000
            min = (; x = -1*ones(n), u = -1*ones(m))
            max = (; x = 1*ones(n), u = 1*ones(m))
            xs = 1:d |> Map(i -> sample(min.x, max.x)) |> collect
            us = 1:d |> Map(i -> sample(min.u, max.u)) |> collect
            fs = zip(xs, us) |> MapSplat((x, u) -> target_function(x, u)) |> collect
            data = (; x=xs, u=us, f=fs)
            println("n = $(n), m = $(m), epochs = $(epochs)")
            i_max = 20
            T = 1e-1
            h_array = [64, 64]
            z_array = h_array  # for PICNN
            u_array = vcat(64, z_array...)  # for PICNN; length(u_array) != length(z_array) + 1
            act = Flux.leakyrelu
            α_is = 1:i_max |> Map(i -> Flux.glorot_uniform(n+m)) |> collect
            β_is = 1:i_max |> Map(i -> Flux.glorot_uniform(1)) |> collect
            # generate approximators
            fnn = FNN(n, m, h_array, act)
            ma = MA(α_is, β_is; n=n, m=m)
            lse = LSE(α_is, β_is, T; n=n, m=m)
            pma = PMA(n, m, i_max, h_array, act)
            plse = PLSE(n, m, i_max, T, h_array, act)
            picnn = PICNN(n, m, u_array, z_array, act, act)
            approximators = (;
                             # fnn=fnn,
                             # ma=ma,
                             # lse=lse,
                             # pma=pma,
                             # plse=plse,
                             picnn=picnn,
                            )
            _dir_save = joinpath(__dir_save, "n=$(n)_m=$(m)_epochs=$(epochs)")
            approximators |> Map(approximator -> test_all(approximator, data, epochs, __dir_save)) |> collect
        end
    end
end
