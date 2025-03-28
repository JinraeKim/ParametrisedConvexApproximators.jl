using ParametrisedConvexApproximators
using Flux
# using CUDA
using Plots
using Random
using ParameterSchedulers
using Statistics: mean


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

function loss_nonnegativity_violation(model, x, u, f)
    pred = model(x, u)
    l = mean(Flux.relu(-pred))
    return l
end


function main(epochs=2)
    pcm = PLSE(n, m, i_max, T, h_array, act)
    nn = FNN(n, m, h_array, act)
    model = LooslyCoupledModel(pcm, nn)

    # target_function = example_target_function(:quadratic_sin_sum)
    target_function = (x, u) -> x[1]^2 + (u[1]^4 - u[1]^2)
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
    ls_nonnegativity_violation = []
    """
    Argument `network` is necessary. This will be converted into cpu by default.
    """
    function callback(network, epoch)
        # @show l_mse = get_loss(network, dataset[:test], loss_mse)
        # @show l_minorant = get_loss(network, dataset[:test], loss_minorant)
        # @show l_minorant_true = get_loss(network.pcm, dataset[:test], loss_minorant_true)
        @show l_total = get_loss(network, dataset[:test], composite_loss_new)
        @show l_nonnegativity_violation = get_loss(network.nn, dataset[:test], loss_nonnegativity_violation)
        # push!(ls_mse, l_mse)
        # push!(ls_minorant, l_minorant)
        # push!(ls_minorant_true, l_minorant_true)
        push!(ls_total, l_total)
        push!(ls_nonnegativity_violation, l_nonnegativity_violation)
        c_plot = range(min_condition[1], stop=max_condition[1]; length=100)
        d_plot = range(min_decision[1], stop=max_decision[1]; length=100)
        # 3d plot
        fig_vis1 = plot(; title="network", xlabel="c", ylabel="d")
        fig_vis2 = plot(; title="pcm", xlabel="c", ylabel="d")
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> network([c], [d])[1]; st=:surface, alpha=0.5)
        plot!(fig_vis2, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(fig_vis2, c_plot, d_plot, (c, d) -> network.pcm([c], [d])[1]; st=:surface, alpha=0.5)
        fig_vis = plot(fig_vis1, fig_vis2; layout=(2, 1))
        # contour
        fig_ctr = plot(; title="contour", xlabel="c", ylabel="d")
        plot!(fig_ctr, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:contour, alpha=0.5)
        cs_ctr = -1:0.1:1
        plot!(fig_ctr, cs_ctr, hcat([minimise(network.pcm, [c]) for c in cs_ctr]...)'; label="optimal from pcm")
        # loss
        fig_loss = plot(;
            ylabel="Test loss",
            ylim=(-0.5, 2.5),
        )
        plot!(fig_loss, 1:length(ls_mse), ls_mse; label="MSE")
        plot!(fig_loss, 1:length(ls_minorant), ls_minorant; label="Minorant (loss)")
        plot!(fig_loss, 1:length(ls_minorant), ls_minorant_true; label="Minorant (true)")
        plot!(fig_loss, 1:length(ls_nonnegativity_violation), ls_nonnegativity_violation; label="Nonnegativity violation")
        plot!(fig_loss, 1:length(ls_total), ls_total; label="Total")
        fig_ctr_loss = plot(fig_ctr, fig_loss; layout=(2, 1))
        # total
        fig = plot(fig_vis, fig_ctr_loss; layout=(1, 2))
        # frame(anim)
        display(fig)
    end
    Flux.train!(
        trainer;
        # batchsize=1024,
        batchsize=16,
        epochs,
        callback,
        # device=gpu,
        device=cpu,
    )
    # gif(anim, "composite_loss_for_pcm.gif", fps=10)
end
