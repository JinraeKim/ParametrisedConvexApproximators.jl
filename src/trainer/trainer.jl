abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    network::AbstractApproximator
    dataset::DecisionMakingDataset
    loss
    optimiser
    function SupervisedLearningTrainer(
        dataset, network;
        normalisation=nothing,
        loss=Flux.Losses.mse,
        optimiser=Adam(1e-3),
    )
        network = retrieve_normalised_network(network, dataset, normalisation)
        @assert dataset.split == :full
        new(network, dataset, loss, optimiser)
    end
end


function retrieve_normalised_network(network::AbstractApproximator, dataset::DecisionMakingDataset, normalisation)
    if normalisation == nothing
        normalised_network = network
    elseif normalisation == :max_abs
        normalised_network = MaxAbsNormalisedApproximator(network, dataset)
    else
        error("Invalid normalisation method $(normalisation)")
    end
    return normalised_network
end


"""
You must explicitly give "the network to be evaluated".
"""
function get_loss(network, dataset, loss)
    l = loss(network(hcat(dataset.conditions...), hcat(dataset.decisions...)), hcat(dataset.costs...))
    return l
end


function Flux.train!(
        trainer::SupervisedLearningTrainer;
        batchsize=16,
        epochs=200,
        fig_name="loss.pdf",
    )
    (; network, dataset, loss, optimiser) = trainer
    data_train = Flux.DataLoader((
        hcat(dataset[:train].conditions...),
        hcat(dataset[:train].decisions...),
        hcat(dataset[:train].costs...),
    ); batchsize=batchsize)
    opt_state = Flux.setup(optimiser, network)

    losses_train = []
    losses_validate = []
    loss_train = nothing
    loss_validate = nothing
    minimum_loss_validate = Inf
    best_network = nothing
    for epoch in 0:epochs
        println("epoch: $(epoch)/$(epochs)")
        if epoch != 0
            for (x, u, f) in data_train
                val, grads = Flux.withgradient(network) do _network
                    pred = _network(x, u)
                    loss(pred, f)
                end
                # TODO
                Flux.update!(opt_state, network, grads[1])
                if typeof(network) == PICNN
                    project_nonnegative!(network)
                end
            end
        end
        loss_train = get_loss(trainer.network, trainer.dataset[:train], trainer.loss)
        push!(losses_train, loss_train)
        loss_validate = get_loss(trainer.network, trainer.dataset[:validate], trainer.loss)
        push!(losses_validate, loss_validate)
        @show loss_train
        @show loss_validate
        if loss_validate < minimum_loss_validate
            println("Best network found!")
            minimum_loss_validate = loss_validate
            @show minimum_loss_validate
            best_network = deepcopy(network)
        end
    end
    # Deprecated to remove the dependency of Plots.jl
    # # plot
    # if fig_name != nothing
    #     p = plot(; xlabel="epoch", ylabel="loss", yaxis=:log)
    #     plot!(p, 0:epochs, losses_train, label="train")
    #     plot!(p, 0:epochs, losses_validate, label="validate")
    #     savefig(p, fig_name)
    # end
    return best_network
end
