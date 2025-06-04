using ParametrisedConvexApproximators
using Flux
# using CUDA
using Plots
using Random
using ParameterSchedulers
using Statistics: mean
using ForwardDiff
using NLopt


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


## NLopt
function autodiff(f::Function)
    function nlopt_fn(x::Vector, grad::Vector)
        if length(grad) > 0
            # Use ForwardDiff to compute the gradient. Replace with your
            # favorite Julia automatic differentiation package.
            ForwardDiff.gradient!(grad, f, x)
        end
        return f(x)
    end
end

function minimise_nlopt(opt, my_obj_fn, min_decision, max_decision, initial_guess)
    NLopt.maxeval!(opt, 100)  # local opt

    NLopt.lower_bounds!(opt, min_decision)
    NLopt.upper_bounds!(opt, max_decision)
    NLopt.min_objective!(opt, autodiff(my_obj_fn))
    J, min_us, ret = NLopt.optimize(opt, initial_guess)
    return min_us
end


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
    return l
end

function loss_minorant(model, x, u, f)
    # ReLU-like approach
    pred = model(x, u)
    l = mean(Flux.relu(pred .- f))
    return l
end

function composite_loss(model, x, u, f)
    l = 0.0
    l += loss_mse(model, x, u, f)
    l += mean(-model(x, u))
    l += 100 * loss_minorant(model, x, u, f)
    return l
end

function composite_loss_new(model::LooslyCoupledModel, x, u, f)
    (; pcm, nn) = model
    pred_pcm = pcm(x, u)
    pred_gap = nn(x, u)
    pred = pred_pcm + pred_gap
    l_minorant = 100.0 * loss_minorant(pcm, x, u, f)
    l_mse = 1.0 * Flux.Losses.mse(pred, f)
    # l_tight_gap = 0.05 * mean(Flux.mse(pred_gap, 0))
    l_tight_gap = 10.00 * mean(Flux.mae(pred_gap, 0))
    return l_minorant + l_mse + l_tight_gap
end

function loss_nonnegativity_violation(model, x, u, f)
    pred = model(x, u)
    l = mean(Flux.relu(-pred))
    return l
end


"""
model_name: :eplse or :cplse (Extended PLSE or Composite PLSE)
func_name: :symm or :asymm (symmetric or asymmetric)
"""
function main(epochs=2; model_name=:cplse, gen_anim=true, func_name=:asymm, lr=1e-3)
    @show model_name
    @show func_name
    Random.seed!(2025)
    pcm = PLSE(n, m, i_max, T, h_array, act)
    nn = FNN(n, m, h_array, act)
    if model_name == :eplse
        model = EPLSE(pcm, nn, min_decision, max_decision)
    elseif model_name == :cplse
        model = LooslyCoupledModel(pcm, nn)
    elseif model_name == :plse
        model = pcm
    else
        error("Invalid model")
    end

    if func_name == :asymm
        target_function = example_target_function(:quadratic_sin_sum)
    elseif func_name == :symm
        target_function = (x, u) -> x[1]^2 + (u[1]^4 - u[1]^2)
    end
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
    if model_name == :eplse
        loss = loss_mse
    elseif model_name == :cplse
        loss = composite_loss_new
    elseif model_name == :plse
        loss = composite_loss
    else
        error("Invalid model")
    end
    trainer = SupervisedLearningTrainer(
        dataset, model;
        loss,
        optimiser=Flux.Adam(lr),
        # scheduler=ParameterSchedulers.Exp(start=1e-3, decay=0.99),
    )

    if gen_anim
        anim = Animation()
    end

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
        if model_name == :eplse
            @show l_mse = get_loss(network, dataset[:test], loss_mse)
        elseif model_name == :cplse
            @show l_mse = get_loss(network, dataset[:test], loss_mse)
            @show l_minorant = get_loss(network.pcm, dataset[:test], loss_minorant)
            @show l_total = get_loss(network, dataset[:test], composite_loss_new)
            @show l_nonnegativity_violation = get_loss(network.nn, dataset[:test], loss_nonnegativity_violation)
        elseif model_name == :plse
            @show l_mse = get_loss(network, dataset[:test], loss_mse)
            @show l_minorant = get_loss(network, dataset[:test], loss_minorant)
            @show l_total = get_loss(network, dataset[:test], composite_loss)
        end
        if model_name == :eplse
            push!(ls_total, l_mse)
        elseif model_name == :cplse
            push!(ls_mse, l_mse)
            push!(ls_total, l_total)
            push!(ls_minorant, l_minorant)
            push!(ls_nonnegativity_violation, l_nonnegativity_violation)
        elseif model_name == :plse
            push!(ls_mse, l_mse)
            push!(ls_minorant, l_minorant)
            push!(ls_total, l_total)
        end
        c_plot = range(min_condition[1], stop=max_condition[1]; length=100)
        d_plot = range(min_decision[1], stop=max_decision[1]; length=100)
        # 3d plot
        fig_vis1 = plot(; title="network: $(model_name)", xlabel="c", ylabel="d")
        fig_vis2 = plot(; title="pcm", xlabel="c", ylabel="d")
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        plot!(fig_vis1, c_plot, d_plot, (c, d) -> network([c], [d])[1]; st=:surface, alpha=0.5)
        plot!(fig_vis2, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:surface, alpha=0.5)
        if model_name == :eplse
            plot!(fig_vis2, c_plot, d_plot, (c, d) -> network.plse([c], [d])[1]; st=:surface, alpha=0.5)
        elseif model_name == :cplse
            plot!(fig_vis2, c_plot, d_plot, (c, d) -> network.pcm([c], [d])[1]; st=:surface, alpha=0.5)
        elseif model_name == :plse
            plot!(fig_vis2, c_plot, d_plot, (c, d) -> network([c], [d])[1]; st=:surface, alpha=0.5)
        end
        fig_vis = plot(fig_vis1, fig_vis2; layout=(2, 1))
        # contour
        fig_ctr = plot(; title="contour (iter: $epoch/$epochs)", xlabel="c", ylabel="d")
        plot!(fig_ctr, c_plot, d_plot, (c, d) -> target_function([c], [d]); st=:contour, alpha=0.5)
        cs_ctr = -1:0.1:1

        ## NLopt
        opt = NLopt.Opt(:LD_SLSQP, m)
        NLopt.srand(2025)  # to make it deterministic

        if model_name == :eplse
            plot!(fig_ctr, cs_ctr, hcat([minimise(network, [c]; min_decision, max_decision) for c in cs_ctr]...)'; label="solution by pcm")
        elseif model_name == :cplse
            # plot!(fig_ctr, cs_ctr, hcat([minimise(network.pcm, [c]; min_decision, max_decision) for c in cs_ctr]...)'; label="solution by pcm")
            println("NOTE: These are refined solutions with postprocessing...")
            initial_guesses = [minimise(network.pcm, [c]; min_decision, max_decision) for c in cs_ctr]
            plot!(fig_ctr, cs_ctr, hcat(initial_guesses...)'; lw=2.0, label="solution by pcm (initial guess)")
            plot!(fig_ctr, cs_ctr, hcat([minimise_nlopt(opt, d -> target_function([c], d), min_decision, max_decision, initial_guess) for (c, initial_guess) in zip(cs_ctr, initial_guesses)]...)'; lw=2.0, label="solution with NLopt")
        elseif model_name == :plse
            plot!(fig_ctr, cs_ctr, hcat([minimise(network, [c]; min_decision, max_decision) for c in cs_ctr]...)'; label="solution by pcm")
        end
        # loss
        fig_loss = plot(;
            ylabel="Test loss",
            ylim=(-0.5, 2.5),
        )
        if model_name == :eplse
            plot!(fig_loss, 1:length(ls_mse), ls_mse; label="MSE")
        elseif model_name == :cplse
            plot!(fig_loss, 1:length(ls_mse), ls_mse; label="MSE")
            plot!(fig_loss, 1:length(ls_minorant), ls_minorant; label="Minorant (loss)")
            # plot!(fig_loss, 1:length(ls_minorant), ls_minorant_true; label="Minorant (true)")
            plot!(fig_loss, 1:length(ls_nonnegativity_violation), ls_nonnegativity_violation; label="Nonnegativity violation")
            plot!(fig_loss, 1:length(ls_total), ls_total; label="Total")
        elseif model_name == :plse
            plot!(fig_loss, 1:length(ls_mse), ls_mse; label="MSE")
            plot!(fig_loss, 1:length(ls_minorant), ls_minorant; label="Minorant")
            plot!(fig_loss, 1:length(ls_total), ls_total; label="Total")
        end
        fig_ctr_loss = plot(fig_ctr, fig_loss; layout=(2, 1))
        # total
        fig = plot(fig_vis, fig_ctr_loss; layout=(1, 2))
        if gen_anim
            frame(anim)
        end
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
    if gen_anim
        gif(anim, "composite_loss_for_pcm.gif", fps=10)
    end
end
