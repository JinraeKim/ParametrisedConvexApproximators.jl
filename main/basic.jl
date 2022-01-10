using Test
using Flux
using Flux: DataLoader
using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using UnPack
using Transducers
using Convex
using Random
using Plots, StatsPlots
ENV["GKSwstype"]="nul"  # deactivate X server needs
using DataFrames
using Statistics
using LinearAlgebra
using SCS


__dir_save = "main/basic"


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
    println("optimise test; with $(d) test data")
    if typeof(approximator) <: ParametrisedConvexApproximator
        x = rand(n)
        u = Convex.Variable(m)
        @test approximator(x, u) |> size == (1,)  # inference with Convex.jl
    end
    _xs = hcat(data.x...)
    # begin  # TODO: remove it; only for test
    #     u = Convex.Variable(m)
    #     problem = minimize(approximator(_xs[:, 1], u)[1])
    #     problem.constraints += [u >= min_nt.u]
    #     problem.constraints += [u <= max_nt.u]
    #     @time solve!(problem, opt; verbose=true, silent_solver=false)
    # end
    println("Optimise $(d) points")
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
    minimisers_diff_norm = skipmissing(1:d |> Map(i -> minimisers_estimated[i] == [nothing] ? missing : norm(minimisers_estimated[i] - minimisers_true[i])) |> collect)  # make each element 'Number'
    optvals_diff_abs = skipmissing(1:d |> Map(i -> optvals_estimated[i] == [-Inf] || optvals_estimated[i] == [Inf] ? missing : abs((optvals_estimated[i] - optvals_true[i])[1])) |> collect)  # make each element 'Number'
    println("norm(estimated minimiser - true minimiser)'s mean (only for success cases): $(mean(minimisers_diff_norm))")
    println("norm(estimated optval - true optval)'s mean (only for success cases): $(mean(optvals_diff_abs))")
    _result = (;
               benchmark = bchmkr,
               minimisers_diff_norm=minimisers_diff_norm,
               optvals_diff_abs=optvals_diff_abs,
              )
    return _result
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
    _result = optimise_test(approximator, data_test, min_nt, max_nt)
    # figures
    fig_surface = nothing
    @unpack n, m = approximator
    approx_type = approximator_type(approximator)
    if n == 1 && m == 1
        fig_surface = plot_approximator(approximator, min_nt, max_nt)
    else
        println("plotting surface is ignored for high-dimensional cases")
        fig_surface = plot()
    end
    result = (; _result..., fig_surface=fig_surface, approx_type=approximator_type(approximator))
end

function plot_approximator(approximator, min_nt, max_nt; kwargs...)
    l = 20
    _xs = range(min_nt.x[1], stop=max_nt.x[1], length=l)
    _us = range(min_nt.u[1], stop=max_nt.u[1], length=l)
    _f(x, u) = approximator([x], [u])[1]
    fig = plot(_xs, _us, _f;
               st=:wireframe,
               xlim=(min_nt.x[1], max_nt.x[1]),
               ylim=(min_nt.u[1], max_nt.u[1]),
               zlim=(-1.0, 1.0),
               xticks=-0.5:0.5:1.0,
               aspect_ratio=:equal,
               camera=(60, 45),
               xlabel="x",
               ylabel="u",
               colorbar=false,
               xtickfontsize=13, 
               ytickfontsize=13, 
               ztickfontsize=13, 
               xguidefontsize=13, 
               yguidefontsize=13, 
               zguidefontsize=13, 
               legendfontsize=13,
               kwargs...
              )
    fig
end

"""
# References
[1] T. Bian and Z.-P. Jiang, “Value Iteration, Adaptive Dynamic Programming, and Optimal Control of Nonlinear Systems,” in 2016 IEEE 55th Conference on Decision and Control (CDC), Las Vegas, NV, USA, Dec. 2016, pp. 3375–3380. doi: 10.1109/CDC.2016.7798777.
[2] B. L. Stevens, F. L. Lewis, and E. N. Johnson, Aircraft Control and Simulation: Dynamics, Controls Design, and Autonomous Systems, Third edition. Hoboken, N.J: John Wiley & Sons, 2016.
[3] gym, "Humanoid-v2", https://gym.openai.com/envs/Humanoid-v2/
# dimensions
# n/m = 1/1: for visualisation, 13/4: [2, p.185, Table 3.5-2, F-16 Model Test Case], 376/17: [3]
"""

# @testset "basic" begin
# end

function main(n, m; epochs=100, seed=2021)
    @warn("Note: if you change the target function, you may have to change the true minimisers manually (in the function `optimise_test`).")
    df = DataFrame()
    # tests
    Random.seed!(seed)
    # training data
    d = 5_000
    println("No. of data points: $(d)")
    min_nt = (; x = -1*ones(n), u = -1*ones(m))
    max_nt = (; x = 1*ones(n), u = 1*ones(m))
    xs = 1:d |> Map(i -> sample(min_nt.x, max_nt.x)) |> collect
    us = 1:d |> Map(i -> sample(min_nt.u, max_nt.u)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> target_function(x, u)) |> collect
    data = (; x=xs, u=us, f=fs)
    println("#"^10 * " " * "n = $(n), m = $(m), epochs = $(epochs)" * " " * "#"^10)
    i_max = 30
    T = 1e-1
    h_array = [64, 64]  # fix it
    z_array = [64, 64]  # for PICNN
    h_array_fnn = [64, 64]
    multiplication_factor = 1
    u_array_0 = 64
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
        @unpack benchmark, minimisers_diff_norm, optvals_diff_abs, = result
        push!(df, (;
                   n=n, m=m, epochs=epochs,
                   approximator=approximator_type(approximator),
                   optimise_time_mean=benchmark |> mean,
                   minimisers_diff_norm_mean=minimisers_diff_norm |> mean,
                   optvals_diff_abs_mean=optvals_diff_abs |> mean,
                   no_of_minimiser_success=minimisers_diff_norm |> collect |> length,
                   no_of_optval_success=optvals_diff_abs |> collect |> length,
                   number_of_parameters=approximator |> number_of_parameters,
                  );
              cols=:union,
              promote=true,
             )
    end
    # violin plots
    minimisers_diff_norm = results |> Map(result -> result.minimisers_diff_norm) |> collect  # skipmissing's
    optvals_diff_abs = results |> Map(result -> result.optvals_diff_abs) |> collect  # skipmissing's
    approx_types = approximators |> Map(approximator_type) |> collect
    box_minimiser_diff_norm = plot()
    for (approx_type, minimiser_diff_norm) in zip(approx_types, minimisers_diff_norm)
        violin!(box_minimiser_diff_norm,
                [approx_type], minimiser_diff_norm |> collect;  # remove missing's
                legend=nothing,
                xtickfontsize=13, 
                ytickfontsize=13, 
                xguidefontsize=13, 
                yguidefontsize=13, 
                legendfontsize=13,
               )
    end
    savefig(box_minimiser_diff_norm, joinpath(_dir_save, "minimiser_diff_norm.png"))
    savefig(box_minimiser_diff_norm, joinpath(_dir_save, "minimiser_diff_norm.pdf"))
    box_optval_diff_abs = plot()
    for (approx_type, optval_diff_abs) in zip(approx_types, optvals_diff_abs)
        violin!(box_optval_diff_abs,
                [approx_type], optval_diff_abs |> collect;  # remove missing's
                legend=nothing,
                xtickfontsize=13, 
                ytickfontsize=13, 
                xguidefontsize=13, 
                yguidefontsize=13, 
                legendfontsize=13,
               )
    end
    savefig(box_optval_diff_abs, joinpath(_dir_save, "optval_diff_abs.png"))
    savefig(box_optval_diff_abs, joinpath(_dir_save, "optval_diff_abs.pdf"))
    # plotting
    if n == 1 && m == 1
        for result in results
            @unpack fig_surface, approx_type = result
            savefig(fig_surface, joinpath(_dir_save, "surface_" * approx_type * ".png"))
            savefig(fig_surface, joinpath(_dir_save, "surface_" * approx_type * ".pdf"))
        end
        fig_surface_true = plot_approximator(target_function, min_nt, max_nt)
        savefig(fig_surface_true, joinpath(_dir_save, "surface_true.png"))
        savefig(fig_surface_true, joinpath(_dir_save, "surface_true.pdf"))
    end
    title_minimiser_diff_norm = plot(title="2-norm of minimiser errors",
                                     framestyle=nothing,showaxis=false,xticks=false,yticks=false,margin=0Plots.px,
                                    )
    @show df
end
