using Test
using ParametrisedConvexApproximators
using Flux


n, m = 3, 2
i_max = 20
T = 1e-0
h_array = [64, 64]
act = Flux.leakyrelu
d = 3


function main()
    network = PLSE(n, m, i_max, T, h_array, act)
    # _xs = rand(n, d)
    x = rand(n)
    u = minimise(network, x)
    # for i in 1:d
        # u = _us[:, i]
        # grad = Flux.Zygote.gradient(sum, Flux.params(network)[1])
        """
        HOW TO TEST...? Using the minimiser, train the network, and see the change of the parameters of the network...?
        """
        @show grad
        @test all(size(grad) == size(u))
        error("TODO")
    # end
end


@testset "implicit_diff" begin
    main()
end
