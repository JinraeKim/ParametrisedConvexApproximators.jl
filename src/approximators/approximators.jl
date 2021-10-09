abstract type AbstractApproximator end

function infer!(approx::AbstractApproximator, x, u)
    error("Define infer! for type $(typeof(approx))")
end


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


struct NormalisedApproximator <: AbstractApproximator
    approximator::AbstractApproximator
    normaliser::AbstractNormaliser
end

"""
x ∈ ℝ^n or ℝ^(n×d)
u ∈ ℝ^m or ℝ^(m×d)
"""
function (normalised_approximator::NormalisedApproximator)(x, u; output_normalisation=false)
    @unpack approximator, normaliser = normalised_approximator
    x_normal = normalise(normaliser, x, :x)
    u_normal = normalise(normaliser, u, :u)
    f_normal = approximator(x_normal, u_normal)
    if output_normalisation
        f = f_normal
    else
        f = unnormalise(normaliser, f_normal, :f)
    end
    f
end


# approximators
include("FNN.jl")
include("parametrised_convex_approximators/parametrised_convex_approximators.jl")
include("convex_approximators/convex_approximators.jl")
include("flux_params.jl")

