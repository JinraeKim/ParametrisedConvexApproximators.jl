using Test
using ParametrisedConvexApproximators


n, m = 3, 2
N = 1_000
seed = 2022
min_condition = -ones(n)
max_condition = -ones(n)
min_decision = +ones(m)
max_decision = + ones(m)


function test_SimpleDataset(func_name, split)
    dataset = SimpleDataset(
        func_name, split;
        N=N, n=n, m=m, seed=seed,
        min_condition=min_condition,
        max_condition=max_condition,
        min_decision=min_decision,
        max_decision=max_decision,
   )
end


function main()
    for func_name in [:quadratic]
        for split in [:train, :validate, :test]
            test_SimpleDataset(func_name, split)
        end
    end
end


@testset "dataset" begin
    main()
end
