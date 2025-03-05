using ParametrisedConvexApproximators
using Flux
using Plots
using Random
using ParameterSchedulers
using Statistics: mean
# using CUDA


seed = 2022
n, m = 1, 1
N = 5_000
h_array = [64, 64]
act = Flux.leakyrelu
i_max = 20
T = 1.0
# dataset
min_condition = -1 * ones(n)
max_condition = +1 * ones(n)
min_decision = -1 * ones(m)
max_decision = +1 * ones(m)


struct LooslyCoupledModel <: AbstractApproximator
    pcm
    nn
end


function (model::LooslyCoupledModel)(x, u)
    (; pcm, nn) = model
    pred_pcm = pcm(x, u)
    # pred_gap = Flux.leakyrelu(nn(x, u) .- 0)
    pred_gap = nn(x, u)
    return pred_pcm + pred_gap
end


function loss_mse(model, x, u, f)
    pred = model(x, u)
    l = Flux.Losses.mse(pred, f)
    # l = Flux.Losses.mae(pred, f)
    return l
end

function loss_minorant(model, x, u, f)
    # ReLU-like approach
    pred = model(x, u)
    l = 50 * mean(Flux.relu(pred .- f))
    # l = 1000 * mean(Flux.relu(pred .- f) .* 2)

    # LeakyReLU approach
    # l = 100 * mean(Flux.leakyrelu(pred .- f, 0.001))
    # l = 100 * mean(Flux.leakyrelu(pred .- f, 0.001) .* 2)
    # l = 10 * mean(Flux.gelu(pred .- f))
    # l = 10 * mean(Flux.mish(pred .- f))
    return l
end

function loss_minorant_true(model, x, u, f)
    pred = model(x, u)
    return mean(Flux.relu(pred .- f))
end

function composite_loss(model, x, u, f)
    l = 0.0
    l += loss_mse(model, x, u, f)
    l += loss_minorant(model, x, u, f)
    return l
end

function composite_loss_new(model::LooslyCoupledModel, x, u, f)
    (; pcm, nn) = model
    pred_pcm = pcm(x, u)
    pred_gap = nn(x, u)
    pred = pred_pcm + pred_gap
    l_minorant = 100.0 * mean(Flux.relu(pred_pcm .- f))
    l_mse = 100.0 * Flux.Losses.mse(pred, f)
    l_tight_gap = 1.0 * mean(Flux.mse(pred_gap, 0))
    return l_minorant + l_mse + l_tight_gap
end


function main(epochs=2)
    pcm = PLSE(n, m, i_max, T, h_array, act)
    nn = FNN(n, m, h_array, act)
    model = LooslyCoupledModel(pcm, nn)

    target_function = example_target_function(:quadratic_sin_sum)
    conditions, decisions, costs, metadata = generate_dataset(
        target_function;
        N,
        min_condition,
        max_condition,
        min_decision,
        max_decision,
        rng=Xoshiro(1),
    )
    dataset = DecisionMakingDataset(
        conditions, decisions, costs;
        metadata, rng=Xoshiro(2),
        ratio1=0.7, ratio2=0.2,
    )
    trainer = SupervisedLearningTrainer(
        dataset, model;
        loss=composite_loss_new,
        optimiser=Flux.Adam(1e-3),
        # scheduler=ParameterSchedulers.Exp(start=1e-3, decay=0.99),
    )

    anim = Animation()

    ls_mse = []
    ls_minorant = []
    ls_minorant_true = []
    ls_total = []
    function callback(epoch)
        # @show l_mse = get_loss(model, dataset[:test], loss_mse)
        # @show l_minorant = get_loss(model, dataset[:test], loss_minorant)
        # @show l_minorant_true = get_loss(model.pcm, dataset[:test], loss_minorant_true)
        @show l_total = get_loss(model, dataset[:test], composite_loss_new)
        # push!(ls_mse, l_mse)
        # push!(ls_minorant, l_minorant)
        # push!(ls_minorant_true, l_minorant_true)
        push!(ls_total, l_total)
        c_plot = range(min_condition[1], stop=max_condition[1]; length=100)
        d_plot = range(min_decision[1], stop=max_decision[1]; length=100)
        fig_vis1 = plot(; title="model", xlabel="c", ylabel="d")
        fig_vis2 = plot(; title="pcm", xlabel="c", ylabel="d")
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> model([c], [d])[1]; st=:surface, alpha=0.5)
        plot!(fig_vis2, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(fig_vis2, c_plot, d_plot, (c, d) -> model.pcm([c], [d])[1]; st=:surface, alpha=0.5)
        fig_vis = plot(fig_vis1, fig_vis2; layout=(2, 1))
        # frame(anim)
        fig_loss = plot(;
            ylabel="Test loss",
            ylim=(-0.5, 2.5),
        )
        plot!(1:length(ls_mse), ls_mse; label="MSE")
        plot!(1:length(ls_minorant), ls_minorant; label="Minorant (loss)")
        plot!(1:length(ls_minorant), ls_minorant_true; label="Minorant (true)")
        plot!(1:length(ls_total), ls_total; label="Total")
        fig = plot(fig_vis, fig_loss; layout=(1, 2))
        display(fig)
    end
    Flux.train!(
        trainer;
        batchsize=128,
        epochs=200,
        callback,
    )
    # gif(anim, "composite_loss_for_pcm.gif", fps=10)
end
