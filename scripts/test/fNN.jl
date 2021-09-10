using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers


function f(x, u)  # true function
    x'*x + u'*u
end

function main()
    n = 6
    m = 2
    d = 1000
    h_array = [64, 64]
    act = Flux.leakyrelu
    approximator = PCApprox.fNN(n, m, h_array, act)
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    ## training
    xufdata = PCApprox.xufData(xs, us, fs)
    xufdata_train, xufdata_test = PCApprox.partitionTrainTest(xufdata)
    PCApprox.train_approximator!(approximator, xufdata_train, xufdata_test)
end
