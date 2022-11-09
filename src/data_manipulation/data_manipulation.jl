"""
    split_data2(dataset, ratio)

Split a dataset into train and test datasets (array).
"""
function split_data2(dataset, ratio)
    @assert ratio >= 0.0
    @assert ratio <= 1.0
    n = length(dataset)
    idx = Random.shuffle(1:n)
    train_idx = view(idx, 1:floor(Int, ratio*n))
    test_idx = view(idx, (floor(Int, ratio*n)+1):n)
    dataset[train_idx], dataset[test_idx]
end


"""
    split_data3(dataset, ratio1, ratio2)

Split a dataset into train, validate, and test datasets (array).
"""
function split_data3(dataset, ratio1, ratio2)
    @assert ratio1 >= 0.0
    @assert ratio2 >= 0.0
    @assert ratio1 + ratio2 <= 1.0
    dataset_train, dataset_valtest = split_data2(dataset, ratio1)
    dataset_validate, dataset_test = split_data2(dataset_valtest, ratio2/(1.0-ratio1))
    return dataset_train, dataset_validate, dataset_test
end
