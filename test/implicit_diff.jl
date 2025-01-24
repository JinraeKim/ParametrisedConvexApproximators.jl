using Test
using ParametrisedConvexApproximators
using Flux
using ECOS


n, m = 3, 2
i_max = 20
T = 1e-0
h_array = [64, 64]
act = Flux.leakyrelu
min_decision = -2*ones(m)
max_decision = +2*ones(m)


function main(; epochs=2, N=100, N_test=10,)
    model = PLSEplus(n, m, i_max, T, h_array, act)
    X = hcat([2*rand(n) .- 1 for i in 1:N]...)
    U_true = hcat([zeros(m) for i in 1:N]...)
    X_test = hcat([2*rand(n) .- 1 for i in 1:N_test]...)
    U_true_test = hcat([zeros(m) for i in 1:N_test]...)
    opt_state = Flux.setup(Adam(1e-3), model)
    data = Flux.DataLoader((X, U_true), batchsize=16)
    for multithreading in [false, true]
        @show multithreading
        params_init = deepcopy(Flux.trainables(model))
        @time for epoch in 1:epochs
            @show epoch
            @show Flux.Losses.mse(minimise(model, X_test), U_true_test)
            for (x, u_true) in data
                d = size(x)[2]
                val, grads = Flux.withgradient(model) do _model
                    u_star = minimise(_model, x; min_decision, max_decision, multithreading)
                    Flux.Losses.mse(u_star, u_true)
                end
                Flux.update!(opt_state, model, grads[1])
            end
        end
        @test any(Flux.trainables(model) .!= params_init)
    end
end


@testset "implicit_diff" begin
    main()
end
