using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Flux


function main()
    node_array = [1, 2, 3]
    act = Flux.relu
    layer_array = PCApprox.construct_layer_array(node_array, act)
    model = Chain(layer_array...)
    @show model
    nothing
end
