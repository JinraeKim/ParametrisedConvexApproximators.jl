struct PMA <: ParametrisedConvexApproximator
    n::Int
    m::Int
    i_max::Int
    NN::Flux.Chain
    # NN1::Flux.Chain
    # NN2::Flux.Chain
end

"""
Basic constructor PMA based on Flux.Chain.
"""
function PMA(n::Int, m::Int, i_max::Int, h_array::Vector{Int}, act)
    node_array = [n, h_array..., i_max*(m+1)]
    PMA(n, m, i_max, construct_layer_array(node_array, act))
end

"""
    (nn::PMA)(x, u)

Infer using two input arguments `x` and `u`.
# Notes
size(x) = (n, d)
size(u) = (m, d)
# Types
x and u should be arrays.
For example, u = [1] for the one-element case.
"""
function (nn::PMA)(x::AbstractArray, u::AbstractArray)
    is_vector = length(size(x)) == 1
    @assert is_vector == (length(size(u)) == 1)
    x = is_vector ? reshape(x, :, 1) : x
    u = is_vector ? reshape(u, :, 1) : u
    @assert size(x)[2] == size(u)[2]
    tmp = affine_map(nn, x, u)
    _res = maximum(tmp, dims=1)
    res = is_vector ? reshape(_res, 1) : _res
end

"""
When receiving Convex.AbstractExpr input
"""
function (nn::PMA)(x::AbstractArray, u::Convex.AbstractExpr)
    tmp = affine_map(nn, x, u)
    res = [maximum(tmp)]
end

Flux.params(approximator::PMA) = Flux.params(approximator.NN)
