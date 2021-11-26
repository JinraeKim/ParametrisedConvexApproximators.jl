# split data into train and test
"""
    partitionTrainTest(data, at)
Split a dataset into train and test datasets (array).
"""
function partitionTrainTest(data, at)
    if size(data) |> length != 1
        error("Invalid data type")
    end
    n = length(data)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, at*n))
    test_idx = view(idx, (floor(Int, at*n)+1):n)
    data[train_idx], data[test_idx]
end

function partitionTrainTest(data::xufData, at=0.8)
    xs_us_fs = zip(data.x, data.u, data.f) |> collect
    xs_us_fs_train, xs_us_fs_test = partitionTrainTest(xs_us_fs, at)
    # train
    xs_train = xs_us_fs_train |> Map(xuf -> xuf[1]) |> collect
    us_train = xs_us_fs_train |> Map(xuf -> xuf[2]) |> collect
    fs_train = xs_us_fs_train |> Map(xuf -> xuf[3]) |> collect
    data_train = xufData(xs_train, us_train, fs_train)
    # test
    xs_test = xs_us_fs_test |> Map(xuf -> xuf[1]) |> collect
    us_test = xs_us_fs_test |> Map(xuf -> xuf[2]) |> collect
    fs_test = xs_us_fs_test |> Map(xuf -> xuf[3]) |> collect
    data_test = xufData(xs_test, us_test, fs_test)
    data_train, data_test
end

function partitionTrainTest(data::xurx_nextData, at=0.8)
    xs_us_rs_x_nexts = zip(data.x, data.u, data.r, data.x_next) |> collect
    xs_us_rs_x_nexts_train, xs_us_rs_x_nexts_test = partitionTrainTest(xs_us_rs_x_nexts, at)
    # train
    xs_train = xs_us_rs_x_nexts_train |> Map(xurx_next -> xurx_next[1]) |> collect
    us_train = xs_us_rs_x_nexts_train |> Map(xurx_next -> xurx_next[2]) |> collect
    rs_train = xs_us_rs_x_nexts_train |> Map(xurx_next -> xurx_next[3]) |> collect
    x_nexts_train = xs_us_rs_x_nexts_train |> Map(xurx_next -> xurx_next[4]) |> collect
    data_train = xurx_nextData(xs_train, us_train, rs_train, x_nexts_train)
    # test
    xs_test = xs_us_rs_x_nexts_test |> Map(xurx_next -> xurx_next[1]) |> collect
    us_test = xs_us_rs_x_nexts_test |> Map(xurx_next -> xurx_next[2]) |> collect
    rs_test = xs_us_rs_x_nexts_test |> Map(xurx_next -> xurx_next[3]) |> collect
    x_nexts_test = xs_us_rs_x_nexts_test |> Map(xurx_next -> xurx_next[4]) |> collect
    data_test = xurx_nextData(xs_test, us_test, rs_test, x_nexts_test)
    data_train, data_test
end
