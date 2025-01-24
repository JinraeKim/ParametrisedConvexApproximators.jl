struct FNN <: AbstractApproximator
    n::Int
    m::Int
    NN::Flux.Chain
end
Flux.@layer FNN trainable=(NN,)
function FNN(n::Int, m::Int, h_array::Vector{Int}, act)
    node_array = [n+m, h_array..., 1]
    FNN(n, m, construct_layer_array(node_array, act))
end

"""
    (nn::FNN)(x, u)

Infer using two input arguments `x` and `u`.
# Notes
size(x) = (n, d)
size(u) = (m, d)
"""
function (nn::FNN)(x, u)
    res = nn.NN(vcat(x, u))
end
