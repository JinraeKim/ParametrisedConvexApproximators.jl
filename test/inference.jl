using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers


function generate_approximators(n, m)
    # approximators
    h_array = [64, 64]
    act = Flux.leakyrelu
    # fNN
    fnn = PCApprox.fNN(n, m, h_array, act)
    # pMA
    i_max = 20
    pma = PCApprox.pMA(n, m, i_max, h_array, act)
    approximators = (; fNN=fNN, pMA=pMA)
end

function infer(approximator, xu_data)
    @show typeof(approximator)
    xu_nt = PCApprox.Data_to_NamedTuple(xu_data)
    @show approximator(xu_nt.x, xu_nt.u)
end

function main()
    n = 6
    m = 2
    d = 3  # no. of data
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    xu_data = PCApprox.xuData(xs, us)
    approximators = generate_approximators(n, m)
    _ = approximators |> Map(approximator -> infer(approximator, xu_data)) |> collect
    nothing
end
