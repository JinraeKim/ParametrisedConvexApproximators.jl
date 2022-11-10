using Test
using ParametrisedConvexApproximators
using Flux


n, m = 3, 2
i_max = 20
T = 1.0
h_array = [64, 64]
act = Flux.leakyrelu
N = 1_000
seed = 2022
min_condition = -ones(n)
max_condition = -ones(n)
min_decision = +ones(m)
max_decision = +ones(m)


function test_SimpleDataset(func_name, split)
    dataset = SimpleDataset(
        func_name, split;
        N=N, n=n, m=m, seed=seed,
        min_condition=min_condition,
        max_condition=max_condition,
        min_decision=min_decision,
        max_decision=max_decision,
   )
    return dataset
end


function test_SupervisedLearningTrainer(dataset, network; epochs=1)
    trainer = SupervisedLearningTrainer(dataset, network)
    @infiltrate
    for i in 1:epochs
        Flux.train!(trainer)
    end
end


function test_dataset()
    for func_name in [:quadratic, (x, u) -> sum(x)+sum(u)]
        for split in [:train, :validate, :test]
            test_SimpleDataset(func_name, split)
        end
    end
end


function test_trainer()
    dataset = test_SimpleDataset(:quadratic, :train)  # for training
    network = PLSE(n, m, i_max, T, h_array, act)
    test_SupervisedLearningTrainer(dataset, network)
end


function main()
    test_dataset()
    test_trainer()
end


@testset "dataset" begin
    main()
end
