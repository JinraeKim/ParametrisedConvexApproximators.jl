using Test
using Flux
using Flux: DataLoader
using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using UnPack
using Transducers
using Convex
# using BenchmarkTools
using Random
using Plots
ENV["GKSwstype"]="nul"  # deactivate X server needs
using DataFrames
using Statistics


__dir_save = "test/basic"


function target_function(x, u)
    n_x = length(x)
    n_u = length(u)
    [0.5 * ( -(1/n_x)*x'*x + (1/n_u)*u'*u )]
end

function sample(min, max)
    sampled = min + (max - min) .* rand(size(min)...)
end

function approximator_type(approximator)
    approx_type = nothing
    if typeof(approximator) == FNN
        approx_type = "FNN"
    elseif typeof(approximator) == MA
        approx_type = "MA"
    elseif typeof(approximator) == LSE
        approx_type = "LSE"
    elseif typeof(approximator) == PMA
        approx_type = "PMA"
    elseif typeof(approximator) == PLSE
        approx_type = "PLSE"
    elseif typeof(approximator) == PICNN
        approx_type = "PICNN"
    else
        error("Specify save directory")
    end
    approx_type
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

function optimise_test(approximator, data, min_nt, max_nt)
    @unpack n, m = approximator
    d = data.x |> length
    # BenchmarkTools.DEFAULT_PARAMETERS.samples = d  # number of samples
    # BenchmarkTools.DEFAULT_PARAMETERS.seconds = 30  # maximum times
    println("optimise test; with $(d) test data")
    if typeof(approximator) <: ParametrisedConvexApproximator
        x = rand(n)
        u = Convex.Variable(m)
        @test approximator(x, u) |> size == (1,)  # inference with Convex.jl
    end
    _xs = hcat(data.x...)
    # println("Optimise a single point (evaluating via BenchmarkTools)")
    # bchmkr = @benchmark optimise($approximator, sample($(min_nt.u), $(max_nt.u)); u_min=$(min_nt.u), u_max=$(max_nt.u))
    # println("Optimise a single point (analysing the result using BenchmarkTools...)")
    # @btime res = optimise($approximator, rand($n))
    println("Optimise $(d) points")
    # @time res = optimise(approximator, _xs)
    @time res_timed = @timed optimise(approximator, _xs;
                                      u_min=min_nt.u, u_max=max_nt.u,
                                      multithreading=false,
                                     )  # with box constraints
    res = res_timed.value
    bchmkr = res_timed.time / d  # average time
    @test res.minimiser |> size == (m, d)  # optimise; minimiser
    @test res.optval |> size == (1, d)  # optimise; optval
    # compare true and estimated minimisers and optvals
    minimisers_true = 1:d |> Map(i -> zeros(m)) |> collect
    optvals_true = 1:d |> Map(i -> target_function(data.x[i], minimisers_true[i])) |> collect
    minimisers_estimated = 1:d |> Map(i -> res.minimiser[:, i]) |> collect
    optvals_estimated = 1:d |> Map(i -> res.optval[:, i]) |> collect
    minimiser_failure_cases = findall(x -> x == repeat([nothing], m), minimisers_estimated)
    optval_failure_cases = findall(x -> x == [Inf] || x == [-Inf], optvals_estimated)
    if length(minimiser_failure_cases) > 0 || length(optval_failure_cases) > 0
        @warn("(optimisaiton failure) there is at least one failure case to solve the optimisation;
              # of minimiser failure cases (nothing): ($(length(minimiser_failure_cases)) / $(d)),
              # of optval failure cases (-Inf or Inf): ($(length(optval_failure_cases)) / $(d))",
             )
        return nothing, nothing, nothing, missing, missing
    else
        minimisers_diff_norm = 1:d |> Map(i -> norm(minimisers_estimated[i] - minimisers_true[i])) |> collect
        optvals_diff = 1:d |> Map(i -> abs(optvals_estimated[i][1] - optvals_true[i][1])) |> collect
        # println("norm(estimated minimiser - true minimiser)'s mean: $(mean(minimisers_diff_norm))")
        # println("norm(estimated optval - true optval)'s mean: $(mean(optvals_diff))")
        fig_minimiser_diff_norm = histogram(minimisers_diff_norm; label=nothing)
        fig_optval_diff = histogram(optvals_diff; label=nothing)
        return fig_minimiser_diff_norm, fig_optval_diff, bchmkr, minimisers_diff_norm, optvals_diff
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
        # println("epoch: $(epoch) / $(epochs)")
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
        # @show loss(_data_test)
    end
end

function test_all(approximator, data, epochs, min_nt, max_nt)
    # split data
    xs_us_fs = zip(data.x, data.u, data.f) |> collect
    xs_us_fs_train, xs_us_fs_test = partitionTrainTest(xs_us_fs, 0.9)  # 90:10
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
    @show number_of_parameters(approximator)
    # tests
    infer_test(approximator)
    @time training_test(approximator, data_train, data_test, epochs)
    fig_minimiser_diff_norm, fig_optval_diff_abs, bchmkr, minimisers_diff_norm, optvals_diff_abs = optimise_test(approximator, data_test, min_nt, max_nt)
    # figures
    fig_surface = nothing
    @unpack n, m = approximator
    approx_type = approximator_type(approximator)
    if n == 1 && m == 1
        fig_surface = plot_surface(approximator, min_nt, max_nt)
        title!(fig_surface, approx_type)
    else
        println("plotting surface is ignored for high-dimensional cases")
        fig_surface = plot()
    end
    # postprocessing of figures
    if fig_minimiser_diff_norm == nothing
        fig_minimiser_diff_norm = plot()
    end
    if fig_optval_diff_abs == nothing
        fig_optval_diff_abs = plot()
    end
    title!(fig_minimiser_diff_norm, approx_type)
    title!(fig_optval_diff_abs, approx_type)
    (;
     fig_surface=fig_surface,
     fig_minimiser_diff_norm=fig_minimiser_diff_norm,
     fig_optval_diff_abs=fig_optval_diff_abs,
     minimisers_diff_norm=minimisers_diff_norm,
     optvals_diff_abs=optvals_diff_abs,
     benchmark=bchmkr,
    )
end

function plot_surface(approximator, min_nt, max_nt; kwargs...)
    l = 100
    _xs = range(min_nt.x[1], stop=max_nt.x[1], length=l)
    _us = range(min_nt.u[1], stop=max_nt.u[1], length=l)
    _f(x, u) = approximator([x], [u])[1]
    fig = plot(_xs, _us, _f;
               st=:surface,
               xlim=(min_nt.x[1], max_nt.x[1]),
               ylim=(min_nt.u[1], max_nt.u[1]),
               zlim=(-1.0, 1.0),
               camera=(30, 45),
               xlabel="x",
               ylabel="u",
               kwargs...
              )
    fig
end

@testset "basic" begin
    @warn("Note: if you change the target function, you may have to change the true minimisers manually (in the function `optimise_test`).")
    df = DataFrame()
    # tests
    println("TODO: change ns and ms, epochs_list, etc.")
    # ns = [1]
    # ms = [1]
    ns = [1, 5, 20]
    ms = [1, 5, 20]
    for (n, m) in zip(ns, ms)
        # epochs_list = [20]
        epochs_list = [100]
        for epochs in epochs_list
            Random.seed!(2021)
            # training data
            d = 1_000
            println("No. of data points: $(d)")
            min_nt = (; x = -1*ones(n), u = -1*ones(m))
            max_nt = (; x = 1*ones(n), u = 1*ones(m))
            xs = 1:d |> Map(i -> sample(min_nt.x, max_nt.x)) |> collect
            us = 1:d |> Map(i -> sample(min_nt.u, max_nt.u)) |> collect
            fs = zip(xs, us) |> MapSplat((x, u) -> target_function(x, u)) |> collect
            data = (; x=xs, u=us, f=fs)
            println("#"^10 * "n = $(n), m = $(m), epochs = $(epochs)" * "#"^10)
            i_max = 20
            T = 1e-1
            h_array = [32, 32]  # fix it
            z_array = [32, 32]  # for PICNN
            h_array_fnn = nothing
            multiplication_factor = nothing
            u_array_0 = nothing
            if n == 1 && m == 1
                h_array_fnn = [48, 48]
                multiplication_factor = 36
                u_array_0 = 1
            elseif n == 5 && m == 5
                h_array_fnn = [64, 64]
                multiplication_factor = 24
                u_array_0 = 16
            elseif n == 20 && m == 20
                h_array_fnn = [96, 96]
                multiplication_factor = 18
                u_array_0 = 64
            end
            u_array = vcat(u_array_0, z_array...)  # for PICNN; length(u_array) != length(z_array) + 1
            act = Flux.leakyrelu
            α_is = 1:i_max*multiplication_factor |> Map(i -> Flux.glorot_uniform(n+m)) |> collect
            β_is = 1:i_max*multiplication_factor |> Map(i -> Flux.glorot_uniform(1)) |> collect
            # generate approximators
            fnn = FNN(n, m, h_array_fnn, act)
            ma = MA(α_is, β_is; n=n, m=m)
            lse = LSE(α_is, β_is, T; n=n, m=m)
            picnn = PICNN(n, m, u_array, z_array, act, act)
            pma = PMA(n, m, i_max, h_array, act)
            plse = PLSE(n, m, i_max, T, h_array, act)
            approximators = (;
                             fnn=fnn,
                             ma=ma,
                             lse=lse,
                             picnn=picnn,
                             pma=pma,
                             plse=plse,
                            )
            _dir_save = joinpath(__dir_save, "n=$(n)_m=$(m)_epochs=$(epochs)")
            mkpath(_dir_save)
            results = approximators |> Map(approximator -> test_all(approximator, data, epochs, min_nt, max_nt)) |> collect
            for (approximator, result) in zip(approximators, results)
                optimise_time_mean = missing
                minimisers_diff_norm_mean = missing
                optvals_diff_abs_mean = missing
                optimisation_failure = true
                if result.benchmark != nothing
                    optimise_time_mean = mean(result.benchmark)
                    minimisers_diff_norm_mean = mean(result.minimisers_diff_norm)
                    optvals_diff_abs_mean = mean(result.optvals_diff_abs)
                    optimisation_failure = false
                end
                no_params = number_of_parameters(approximator)
                push!(df, (;
                           n=n, m=m, epochs=epochs,
                           approximator=approximator_type(approximator),
                           optimise_time_mean=optimise_time_mean,
                           minimisers_diff_norm_mean=minimisers_diff_norm_mean,
                           optvals_diff_abs_mean=optvals_diff_abs_mean,
                           optimisation_failure=optimisation_failure,
                           number_of_parameters=no_params,
                          );
                      cols=:union,
                      promote=true,
                     )
            end
            # plotting
            if n == 1 && m == 1
                title_surface = plot(title="Trained approximators",
                                     framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0Plots.px,
                                    )
                fig_surface_true = plot_surface(target_function, min_nt, max_nt; xlabel="x", ylabel="u")
                title!(fig_surface_true, "target function")
                savefig(fig_surface_true, joinpath(_dir_save, "surface_true.png"))
                savefig(fig_surface_true, joinpath(_dir_save, "surface_true.pdf"))
                fig_surface = plot(title_surface,
                                   fig_surface_true,
                                   ((results |> Map(result -> result.fig_surface) |> collect)...),
                                   plot(; framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0Plots.px);  # dummy
                                   layout=@layout[a{0.01h}; grid(4, 2)],
                                   size=(800, 1200),
                                  )
                savefig(fig_surface, joinpath(_dir_save, "surface.png"))
                savefig(fig_surface, joinpath(_dir_save, "surface.pdf"))
            end
            title_minimiser_diff_norm = plot(title="2-norm of minimiser errors",
                                             framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0Plots.px,
                                            )
            fig_minimiser_diff_norm = plot(title_minimiser_diff_norm,
                                           (results |> Map(result -> result.fig_minimiser_diff_norm) |> collect)...;
                                           # layout=@layout[a{0.01h}; grid(1, length(approximators))],
                                           layout=@layout[a{0.01h}; grid(3, 2)],
                                           size=(800, 900),
                                          )
            savefig(fig_minimiser_diff_norm, joinpath(_dir_save, "minimiser_diff_norm.png"))
            savefig(fig_minimiser_diff_norm, joinpath(_dir_save, "minimiser_diff_norm.pdf"))
            title_optval_diff_abs = plot(title="Absolute value of optval errors",
                                         framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0Plots.px,
                                        )
            fig_optval_diff_abs = plot(title_optval_diff_abs,
                                       (results |> Map(result -> result.fig_optval_diff_abs) |> collect)...;
                                       # layout=@layout[a{0.01h}; grid(1, length(approximators))],
                                       layout=@layout[a{0.01h}; grid(3, 2)],
                                       size=(800, 900),
                                      )
            savefig(fig_optval_diff_abs, joinpath(_dir_save, "optval_diff_abs.png"))
            savefig(fig_optval_diff_abs, joinpath(_dir_save, "optval_diff_abs.pdf"))
        end
    end
    @show df
end
