using Test
using Flux
using ParameterizedConvexApproximators
using Transducers
using Random


# common parameters
n, m = 3, 2
d = 10
h_array = [64, 64]
z_array = [64, 64]
u_array = vcat(64, z_array...)
act = Flux.leakyrelu
i_max = 20
T = 1.0
u_min = -ones(m)
u_max = ones(m)


function generate_networks()
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
    return networks
end


function test_infer(network; seed=2022)
    println("test_infer")
    Random.seed!(seed)
    x = rand(n)
    u = rand(m)
    f̂ = network(x, u)
    @test size(f̂) == (1,)
end


function test_optimize(network; seed=2022)
    println("test_optimize")
    Random.seed!(seed)
    x = rand(n)
    (; minimizer, optval) = optimize(network, x; u_min=u_min, u_max=u_max)
    @test size(minimizer) == (m,)
    @test size(optval) == (1,)
end


function test_optimize_multiple(network; seed=2022)
    println("test_optimize_multiple")
    Random.seed!(seed)
    x = rand(n, d)
    (; minimizer, optval) = optimize(network, x; u_min=u_min, u_max=u_max)
    @test size(minimizer) == (m, d)
    @test size(optval) == (1, d)
end


function test_infer_multiple(network; seed=2022)
    println("test_infer_multiple")
    Random.seed!(seed)
    x = rand(n, d)
    u = rand(m, d)
    f̂ = network(x, u)
    @test size(f̂) == (1, d)
end


function test_network(network)
    test_infer(network)
    test_infer_multiple(network)
    test_optimize(network)
    test_optimize_multiple(network)
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
