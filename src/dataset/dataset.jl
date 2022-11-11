abstract type DecisionMakingDataset end


struct SimpleDataset <: DecisionMakingDataset
    metadata::NamedTuple
    split::Symbol
    conditions::Array
    decisions::Array
    costs::Array
end

function SimpleDataset(
        func;
        n::Int=1, m::Int=1,
        N::Int=1_000, seed=2022,
        ratio1=0.7, ratio2=0.2,
        min_condition=-ones(n),
        max_condition=+ones(n),
        min_decision=-ones(m),
        max_decision=+ones(m),
    )
    @assert all(min_condition .<= max_condition)
    @assert all(min_decision .<= max_decision)
    # split indicies
    Random.seed!(seed)
    train_idx, validate_idx, test_idx = split_data3(collect(1:N), ratio1, ratio2)
    # get data
    f = target_function(func)
    conditions = sample_from_bounds(N, min_condition, max_condition, seed)
    decisions = sample_from_bounds(N, min_decision, max_decision, seed)
    costs = zip(conditions, decisions) |> MapSplat((x, u) -> f(x, u)) |> collect
    metadata = (;
                target_function=f,
                target_function_name=typeof(func) == Symbol ? func : nothing,
                split_ratio=(;
                             train=ratio1,
                             validate=ratio2,
                             test=1-(ratio1+ratio2),
                            ),
                min_condition=min_condition,
                max_condition=max_condition,
                min_decision=min_decision,
                max_decision=max_decision,
                train_idx=train_idx,
                validate_idx=validate_idx,
                test_idx=test_idx,
               )
    return SimpleDataset(metadata, :full, conditions, decisions, costs)
end



function Base.getindex(dataset::SimpleDataset, split)
    (; metadata, conditions, decisions, costs) = dataset
    @assert split in (:train, :validate, :test, :full)
    if split == :full
        dataset_ = dataset
    else
        if split == :train
            idx = metadata.train_idx
        elseif split == :validate
            idx = metadata.validate_idx
        else
            idx = metadata.test_idx
        end
        dataset_ = SimpleDataset(metadata, split, conditions[idx], decisions[idx], costs[idx])
    end
    return dataset_
end


"""
    split_data2(dataset, ratio)

Split a dataset into train and test datasets (array).
"""
function split_data2(dataset, ratio; seed=2022)
    Random.seed!(seed)
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
function split_data3(dataset, ratio1, ratio2; seed=2022)
    Random.seed!(seed)
    @assert ratio1 >= 0.0
    @assert ratio2 >= 0.0
    @assert ratio1 + ratio2 <= 1.0
    dataset_train, dataset_valtest = split_data2(dataset, ratio1)
    dataset_validate, dataset_test = split_data2(dataset_valtest, ratio2/(1.0-ratio1))
    return dataset_train, dataset_validate, dataset_test
end


function sample_from_bounds(N, min_value, max_value, seed)
    samples = []
    for i in 1:N
        sampled_value = (
             min_value
             + (max_value - min_value) .* rand(size(min_value)...)
        )
        push!(samples, sampled_value)
    end
    return samples
end


function target_function(name)
    if typeof(name) == Symbol
        if name == :quadratic
            func(x::Vector, u::Vector) = transpose(x)*x + transpose(u)*u
        else
            error("Undefined simple function")
        end
    elseif typeof(name) <: Function
        func = name
    else
        error("Invalid target function type $(typeof(name))")
    end
    return func
end
