abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    loss
    network
    optimizer
    dataset
    function SupervisedLearningTrainer(
        dataset, network;
        loss=(x, u, f) -> Flux.mse(network(x, u), f),  # TODO: what agg?
        optimizer=Adam(1e-4),
    )
        @assert dataset.split == :full
        new(loss, network, optimizer, dataset)
    end
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
        throttle_time=5,  # [s]
    )
    (; loss, network, optimizer, dataset) = trainer
    cb = Flux.throttle(throttle_time) do
        println("loss_train: $(get_loss(trainer, :train))")
        println("loss_validate: $(get_loss(trainer, :validate))")
    end
    parameters = Flux.params(network)
    data_train = Flux.DataLoader((
        hcat(dataset[:train].conditions...),
        hcat(dataset[:train].decisions...),
        hcat(dataset[:train].costs...),
    ); batchsize=batchsize)
    Flux.train!(loss, parameters, data_train, optimizer; cb=cb)
end
