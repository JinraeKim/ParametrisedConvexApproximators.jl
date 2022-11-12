using Test
using ParameterizedConvexApproximators
using Flux
# using JLD2, FileIO


n, m = 3, 2
i_max = 20
T = 1e-0
h_array = [64, 64]
act = Flux.leakyrelu
N = 1_000
seed = 2022
min_condition = -ones(n)
max_condition = +ones(n)
min_decision = -ones(m)
max_decision = +ones(m)
ratio1 = 0.7
ratio2 = 0.2


function test_split_data2()
    dataset = []
    for i in 1:N
        push!(dataset, rand(n))
    end
    dataset_train, dataset_test = split_data2(dataset, ratio1)
    @test length(dataset_train) == round(N * ratio1)
    @test length(dataset_test) == round(N * (1-ratio1))
end


function test_split_data3()
    dataset = []
    for i in 1:N
        push!(dataset, rand(n))
    end
    dataset_train, dataset_validate, dataset_test = split_data3(dataset, ratio1, ratio2)
    @test length(dataset_train) == round(N * ratio1)
    @test length(dataset_validate) == round(N * ratio2)
    @test length(dataset_test) == round(N * (1-(ratio1+ratio2)))
end


function test_SimpleDataset(func_name, split)
    dataset = SimpleDataset(
        func_name;
        N=N, n=n, m=m, seed=seed,
        min_condition=min_condition,
        max_condition=max_condition,
        min_decision=min_decision,
        max_decision=max_decision,
   )
    return dataset[split]
end


function test_SupervisedLearningTrainer(dataset, network; epochs=3)
    trainer = SupervisedLearningTrainer(dataset, network)
    @show get_loss(trainer, :train)
    @show get_loss(trainer, :validate)
    best_network = Flux.train!(trainer; epochs=epochs)
    @show get_loss(trainer, :test)
    return best_network
end


function test_dataset()
    for func_name in [
                      :quadratic,
                      :parameterized_convex_basic,
                      :quadratic_sin_sum,
                      (x, u) -> sum(x)+sum(u),  # anonymous function example
                     ]
        for split in [:train, :validate, :test]
            test_SimpleDataset(func_name, split)
        end
    end
end


function test_trainer()
    dataset = test_SimpleDataset(:quadratic, :full)  # for trainer
    # dataset = test_SimpleDataset(:parameterized_convex_basic, :full)  # for trainer
    # dataset = test_SimpleDataset(:quadratic_sin_sum, :full)  # for trainer
    network = PLSE(n, m, i_max, T, h_array, act)
    # network = FNN(n, m, h_array, act)
    best_network = test_SupervisedLearningTrainer(dataset, network)
    # save and load example
    # save("example.jld2"; best_network=best_network, network=network)
    # best_network_ = load("example.jld2")["best_network"]
end


function main()
    test_split_data2()
    test_split_data3()
    test_dataset()
    test_trainer()
end


@testset "dataset" begin
    main()
end
