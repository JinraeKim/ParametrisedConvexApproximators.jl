abstract type AbstractTrainer end


struct SupervisedLearningTrainer <: AbstractTrainer
    loss
    network
    optimizer
    dataset_train
    dataset_validate
    dataset_test
    function SupervisedLearningTrainer(
        dataset_train, dataset_validate, dataset_test, network;
        loss=(x, u, f) -> Flux.mse(network(x, u), f),  # TODO: what agg?
        optimizer=Adam(1e-3),
    )
        new(loss, network, optimizer, dataset_train, dataset_validate, dataset_test)
    end
end


function get_loss(trainer::SupervisedLearningTrainer, split::Symbol)
    @assert split ∈ (:train, :validate, :test)
    (; loss) = trainer
    if split == :train
        dataset = trainer.dataset_train
    elseif split == :validate
        dataset = trainer.dataset_validate
    else split == :test
        dataset = trainer.dataset_test
    end
    l = loss(hcat(dataset.conditions...), hcat(dataset.decisions...), hcat(dataset.costs...))
    return l
end


function Flux.train!(
        trainer::SupervisedLearningTrainer;
        batchsize=16,
        throttle_time=5.0,  # [s]
    )
    (; loss, network, optimizer, dataset_train, dataset_validate) = trainer
    @assert dataset_train.split == :train
    evalcb = function()
        println("loss_train: $(get_loss(trainer, :train))")
        println("loss_validate: $(get_loss(trainer, :validate))")
    end
    cb = Flux.throttle(evalcb, throttle_time)
    parameters = Flux.params(network)
    data_train = Flux.DataLoader((
        hcat(dataset_train.conditions...),
        hcat(dataset_train.decisions...),
        hcat(dataset_train.costs...),
    ); batchsize=batchsize)
    Flux.train!(loss, parameters, data_train, optimizer; cb=cb)
end
