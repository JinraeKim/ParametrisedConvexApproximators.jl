using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux
using Transducers


function f(x, u)  # true function
    # TODO: size restriction!!!
    x'*x + u'*u
end

function main()
    n = 6
    m = 2
    d = 2
    h_array = [64, 64]
    act = Flux.leakyrelu
    approximator = PCApprox.fNN(n, m, h_array, act)
    # infer
    xs = 1:d |> Map(i -> rand(n)) |> collect
    us = 1:d |> Map(i -> rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    ## training
    # train data
    xufdata = PCApprox.xufData(xs, us, fs)
    @bp
    xufdata_train, xufdata_test = PCApprox.partitionTrainTest(xufdata)
    # xs_us_fs = zip(xs, us, fs) |> collect
    # xs_us_fs_train, xs_us_fs_test = PCApprox.partitionTrainTest(xs_us_fs)
    # xs_train = xs_us_fs_train |> Map(xu -> xu[1]) |> collect
    # us_train = xs_us_fs_train |> Map(xu -> xu[2]) |> collect
    # fs_train = xs_us_fs_train |> Map(xu -> xu[3]) |> collect
    # x_train, u_train, f_train = hcat(xs_train...), hcat(us_train...), hcat(fs_train...)  # for Flux
    # data_train_nt = (; x=x_train, u=u_train, f=f_train)
    # test data
    # xs_test = xs_us_fs_test |> Map(xu -> xu[1]) |> collect
    # us_test = xs_us_fs_test |> Map(xu -> xu[2]) |> collect
    # fs_test = xs_us_fs_test |> Map(xu -> xu[3]) |> collect
    # x_test, u_test, f_test = hcat(xs_test...), hcat(us_test...), hcat(fs_test...)  # for Flux
    # data_test_nt = (; x=x_test, u=u_test, f=f_test)
    PCApprox.train_approximator!(approximator, xufdata_train, xufdata_test)
end
