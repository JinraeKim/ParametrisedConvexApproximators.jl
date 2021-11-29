abstract type AbstractApproximator end


function construct_layer_array(node_array, act::AbstractVector)
    layer_array = []
    for i in 2:length(node_array)
        node_prev = node_array[i-1]
        node = node_array[i]
        _act = i == length(node_array) ? Flux.identity : act[i-1]
        push!(layer_array, Dense(node_prev, node, _act))
    end
    model = Chain(layer_array...)
    return model
end


"""
    construct_layer_array(node_array, act; act_terminal=Flux.identity)

A convenient way to generate a layer array.

# Example
node_array = [1, 2, 3]
act = Flux.relu
layer_array = PCApprox.construct_layer_array(node_array, act)
model = Chain(layer_array...)
"""
function construct_layer_array(node_array, act; act_terminal=Flux.identity)
    l = length(node_array)
    act_vec = vcat(repeat([act], l-2), [act_terminal])
    construct_layer_array(node_array, act_vec)
end


include("FNN.jl")
include("parametrised_convex_approximators/parametrised_convex_approximators.jl")
