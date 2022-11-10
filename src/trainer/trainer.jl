abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    loss
    parameters
    optimizer
    dataset
    function SupervisedLearningTrainer(
        dataset, network;
        loss=(x, u, f) -> Flux.mse(network(x, u), f),  # TODO: what agg?
        optimizer=Adam(1e-3),
    )
        parameters = Flux.params(network)
        return new(loss, parameters, optimizer, dataset)
    end
end


function Flux.train!(trainer::SupervisedLearningTrainer; batchsize=16, cb=nothing)
    (; loss, parameters, optimizer, dataset) = trainer
    @assert dataset.split == :train
    data = Flux.DataLoader((dataset.conditions, dataset.decisions, dataset.costs); batchsize=batchsize)
    Flux.train!(loss, parameters, data, optimizer; cb=cb)
end
