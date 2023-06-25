using Test
using Flux
using ParametrisedConvexApproximators
using ParametrisedConvexApproximators: sample_from_bounds
using Transducers
using Random


# common parameters
seed = 2022
n, m = 3, 2
d = 10
h_array = [64, 64]
z_array = [64, 64]
u_array = vcat(64, z_array...)
act = Flux.leakyrelu
i_max = 20
T = 1.0
x_min = -ones(n)
x_max = +ones(n)
u_min = -ones(m)
u_max = +ones(m)
x_max_abs = max.(abs.(x_min), abs.(x_max))
u_max_abs = max.(abs.(u_min), abs.(u_max))
f_max_abs = [1e3]  # a random yet sufficiently large number
xs = hcat(sample_from_bounds(d, x_min, x_max, seed)...)
us = hcat(sample_from_bounds(d, u_min, u_max, seed)...)
initial_guess = us


function generate_networks()
    fnn = FNN(n, m, h_array, act)
    ma = MA(n, m, i_max)
    lse = LSE(n, m, i_max, T)
    picnn = PICNN(n, m, u_array, z_array, act, act)
    pma = PMA(n, m, i_max, h_array, act)
    plse = PLSE(n, m, i_max, T, h_array, act)
    plse_plus = PLSE(n, m, i_max, T, h_array, act; strict=true)
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
                    "PLSEPlus" => plse_plus,
                    "DLSE" => dlse,
                   )
    return networks
end


function test_infer(network)
    println("test_infer")
    x = xs[:, 1]
    u = us[:, 1]
    f̂ = network(x, u)
    @test size(f̂) == (1,)
end


function test_minimise(network)
    println("test_minimise")
    x = xs[:, 1]
    minimiser = minimise(network, x; u_min=u_min, u_max=u_max)
    @test size(minimiser) == (m,)
    @test size(network(x, minimiser)) == (1,)
end


function test_minimise_multiple(network)
    println("test_minimise_multiple")
    minimisers = minimise(network, xs; u_min=u_min, u_max=u_max)
    @test size(minimisers) == (m, d)
    @test size(network(xs, minimisers)) == (1, d)
end


function test_infer_multiple(network)
    println("test_infer_multiple")
    f̂ = network(xs, us)
    @test size(f̂) == (1, d)
end


function test_network(network)
    test_infer(network)
    test_infer_multiple(network)
    test_minimise(network)
    test_minimise_multiple(network)
    test_max_abs_normalised_network(network)
end


function test_max_abs_normalised_network(network)
    println("normalised network")
    normalised_network = MaxAbsNormalisedApproximator(
                             deepcopy(network),
                             x_max_abs,
                             u_max_abs,
                             f_max_abs,
                         )
    # inference
    fs_normalised = network(xs ./ x_max_abs, us ./ u_max_abs)
    fs = normalised_network(xs, us)
    @test fs_normalised .* f_max_abs == fs
    # optimization
    minimiser = minimise(network, xs ./ x_max_abs; u_min=u_min ./ u_max_abs, u_max=u_max ./ u_max_abs, initial_guess=initial_guess ./ u_max_abs)
    minimizer_normalized = minimise(normalised_network, xs; u_min=u_min, u_max=u_max, initial_guess=initial_guess)
    @test minimiser .* u_max_abs ≈ minimizer_normalized
    @test network(xs ./ x_max_abs, minimiser) .* f_max_abs ≈ normalised_network(xs, minimizer_normalized)
end


function main()
    networks = generate_networks()
    for (name, network) in networks
        println("Tested network: " * name)
        test_network(network)
    end
end


@testset "networks" begin
    main()
end
