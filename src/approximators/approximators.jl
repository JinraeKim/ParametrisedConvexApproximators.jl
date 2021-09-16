abstract type AbstractApproximator end
abstract type ParametrisedConvexApproximator <: AbstractApproximator end


"""
    construct_layer_array(node_array, act)

A convenient way to generate a layer array.

# Example
node_array = [1, 2, 3]
act = Flux.relu
layer_array = PCApprox.construct_layer_array(node_array, act)
model = Chain(layer_array...)
"""
function construct_layer_array(node_array, act)
    layer_array = []
    for i in 2:length(node_array)
        node_prev = node_array[i-1]
        node = node_array[i]
        _act = i == length(node_array) ? Flux.identity : act
        push!(layer_array, Dense(node_prev, node, _act))
    end
    model = Chain(layer_array...)
    return model
end


# approximators
include("fNN.jl")
include("pMA.jl")
include("flux_params.jl")
