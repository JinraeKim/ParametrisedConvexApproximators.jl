using Test
using ParametrisedConvexApproximators
using LinearAlgebra
using Flux


function main()
    n, m = 3, 2
    d = 10
    # dataset
    X = rand(n, d)
    Y = rand(m, d)
    Z = hcat([norm(X[:, i])+norm(Y[:, i]) for i in 1:d]...)
    # network construction
    i_max = 20
    T = 1e-0
    h_array = [64, 64]
    act = Flux.leakyrelu
    N = 1_000  # The result may be poor if it's too low
    model = PLSE(n, m, i_max, T, h_array, act)
    params_init = deepcopy(Flux.params(model))
    @test all(Flux.params(model) .== params_init)
    # training
    data = Flux.DataLoader((X, Y, Z), batchsize=32)
    opt_state = Flux.setup(Adam(1e-4), model)
    for epoch in 1:10
        for (x, y, z) in data
            val, grads = Flux.withgradient(model) do m
                pred = m(x, y)
                Flux.Losses.mse(pred, z)
            end
            Flux.update!(opt_state, model, grads[1])
        end
    end
    @test all(Flux.params(model) .!= params_init)
end


@testset "dataset" begin
    main()
end
