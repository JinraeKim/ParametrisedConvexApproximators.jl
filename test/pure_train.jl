using Test
using ParametrisedConvexApproximators
using Flux


seed = 2022
n, m = 3, 2
d = 30
h_array = [64, 64]
z_array = [64, 64]
u_array = vcat(64, z_array...)
act = Flux.leakyrelu
i_max = 20
T = 1.0
# dataset
X = rand(n, d)
Y = rand(m, d)
Z = hcat([sum(X[:, i].^2)+sum(Y[:, i].^2) for i in 1:d]...)
# network construction


function main()
    fnn = FNN(n, m, h_array, act)
    ma = MA(n, m, i_max)
    lse = LSE(n, m, i_max, T)
    picnn = PICNN(n, m, u_array, z_array, act, act)
    pma = PMA(n, m, i_max, h_array, act)
    plse = PLSE(n, m, i_max, T, h_array, act)
    dlse = DLSE(
                LSE(n, m, i_max, T),
                LSE(n, m, i_max, T),
               )

    networks = Dict(
                    "FNN" => fnn,
                    "MA" => ma,
                    "LSE" => lse,
                    "PICNN" => picnn,
                    "PMA" => pma,
                    "PLSE" => plse,
                    "DLSE" => dlse,
                   )

    for (name, model) in networks
        @show name
        params_init = deepcopy(Flux.params(model))
        @test all(Flux.params(model) .== params_init)
        # training
        data = Flux.DataLoader((X, Y, Z), batchsize=16)
        # @infiltrate
        opt_state = Flux.setup(Adam(1e-4), model)
        # @infiltrate
        for epoch in 1:2
            for (x, y, z) in data
                val, grads = Flux.withgradient(model) do m
                    pred = m(x, y)
                    Flux.Losses.mse(pred, z)
                end
                Flux.update!(opt_state, model, grads[1])
                if typeof(model) == PICNN
                    project_nonnegative!(model)
                end
            end
        end
        @test any(Flux.params(model) .!= params_init)
    end
end


@testset "dataset" begin
    main()
end
