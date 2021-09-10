using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers


function f(x, u)  # true function
    -x'*x + u'*u
end

function supervised_learning(approximator, xufdata)
    @show typeof(approximator)
    ## training
    xufdata_train, xufdata_test = PCApprox.partitionTrainTest(xufdata)
    PCApprox.train_approximator!(approximator, xufdata_train, xufdata_test)
end

function main()
    n = 6
    m = 2
    d = 1000  # no. of data
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    xufdata = PCApprox.xufData(xs, us, fs)
    # approximators
    h_array = [64, 64]
    act = Flux.leakyrelu
    approximator = PCApprox.fNN(n, m, h_array, act)
    # training
    supervised_learning(approximator, xufdata)
end
