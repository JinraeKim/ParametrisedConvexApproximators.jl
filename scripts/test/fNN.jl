using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers


function main()
    n = 6
    m = 2
    d = 1000  # no. of data
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    # approximators
    h_array = [64, 64]
    act = Flux.leakyrelu
    approximator = PCApprox.fNN(n, m, h_array, act)
    @show approximator(xs[1], us[1])
end
