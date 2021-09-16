using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers
using Random


function f(x, u)  # true function
    0.5 * (-x'*x + u'*u)
end

function supervised_learning(approximator, xuf_data)
    @show typeof(approximator)
    ## training
    xuf_data_train, xuf_data_test = PCApprox.partitionTrainTest(xuf_data)
    PCApprox.train_approximator!(approximator, xuf_data_train, xuf_data_test)
end

function main(; seed=2021)
    Random.seed!(seed)
    n = 6
    m = 2
    d = 1000  # no. of data
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    xuf_data = PCApprox.xufData(xs, us, fs)
    # approximators
    h_array = [64, 64]
    act = Flux.leakyrelu
    approximator = PCApprox.fNN(n, m, h_array, act)
    # training
    supervised_learning(approximator, xuf_data)
end
