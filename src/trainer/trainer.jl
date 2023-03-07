abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    loss
    network::AbstractApproximator
    optimiser
    dataset::DecisionMakingDataset
    function SupervisedLearningTrainer(
        dataset, network;
        loss=(x, u, f) -> Flux.mse(network(x, u), f),
        optimiser=Adam(1e-3),
        normalisation=:max_abs,
    )
        network = retrieve_normalised_network(network, dataset, normalisation)
        @assert dataset.split == :full
        new(loss, network, optimiser, dataset)
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


function get_loss(trainer::SupervisedLearningTrainer, split::Symbol)
    @assert split ∈ (:train, :validate, :test)
    dataset = trainer.dataset[split]
    (; loss) = trainer
    l = loss(hcat(dataset.conditions...), hcat(dataset.decisions...), hcat(dataset.costs...))
    return l
end


function Flux.train!(
        trainer::SupervisedLearningTrainer;
        batchsize=16,
        epochs=200,
        fig_name="loss.pdf",
    )
    (; loss, network, optimiser, dataset) = trainer
    parameters = Flux.params(network)
    data_train = Flux.DataLoader((
        hcat(dataset[:train].conditions...),
        hcat(dataset[:train].decisions...),
        hcat(dataset[:train].costs...),
    ); batchsize=batchsize)

    losses_train = []
    losses_validate = []
    loss_train = nothing
    loss_validate = nothing
    minimum_loss_validate = Inf
    best_network = nothing
    for epoch in 0:epochs
        println("epoch: $(epoch)/$(epochs)")
        if epoch != 0
            Flux.train!(loss, parameters, data_train, optimiser)
        end
        loss_train = get_loss(trainer, :train)
        push!(losses_train, loss_train)
        loss_validate = get_loss(trainer, :validate)
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
    # plot
    if fig_name != nothing:
        p = plot(; xlabel="epoch", ylabel="loss", yaxis=:log)
        plot!(p, 0:epochs, losses_train, label="train")
        plot!(p, 0:epochs, losses_validate, label="validate")
        savefig(p, fig_name)
    end
    return best_network
end
