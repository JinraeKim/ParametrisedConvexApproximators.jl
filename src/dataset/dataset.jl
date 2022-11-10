abstract type DecisionMakingDataset end


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


struct SimpleDataset <: DecisionMakingDataset
    metadata::NamedTuple
    split::Symbol
    conditions::Array
    decisions::Array
    costs::Array
    function SimpleDataset(
            func,
            split::Symbol;
            n::Int=1, m::Int=1,
            N::Int=1_000, seed=2022,
            ratio1=0.7, ratio2=0.2,
            min_condition=-ones(n),
            max_condition=+ones(n),
            min_decision=-ones(m),
            max_decision=+ones(m),
        )
        @assert split ∈ (:train, :validate, :test)
        @assert all(min_condition .<= max_condition)
        @assert all(min_decision .<= max_decision)
        # split indicies
        Random.seed!(seed)
        train_idx, validate_idx, test_idx = split_data3(collect(1:N), ratio1, ratio2)
        if split == :train
            idx = train_idx
        elseif split == :validate
            idx = validate_idx
        else
            idx = test_idx
        end
        # get data
        f = target_function(func)
        conditions = sample_from_bounds(N, min_condition, max_condition, seed)[idx]
        decisions = sample_from_bounds(N, min_decision, max_decision, seed)[idx]
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
                   )
        new(metadata, split, conditions, decisions, costs)
    end
end
