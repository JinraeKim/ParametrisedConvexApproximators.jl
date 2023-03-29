struct PLSE <: ParametrisedConvexApproximator
    n::Int
    m::Int
    i_max::Int
    T::Real
    NN::Flux.Chain
    strict::Bool
end
Flux.@functor PLSE (NN,)
function PLSE(n::Int, m::Int, i_max::Int, T::Real, h_array::Vector{Int}, act; strict=false)
    @assert T > 0
    node_array = [n, h_array..., i_max*(m+1)]
    PLSE(n, m, i_max, T, construct_layer_array(node_array, act), strict)
end


"""
    (nn::PLSE)(x, u)

Infer using two input arguments `x` and `u`.
# Notes
size(x) = (n, d)
size(u) = (m, d)
# Types
x and u should be arrays.
For example, u = [1] for the one-element case.
"""
function (nn::PLSE)(x::AbstractArray, u::AbstractArray)
    (; T) = nn
    is_vector = length(size(x)) == 1
    @assert is_vector == (length(size(u)) == 1)
    x = is_vector ? reshape(x, :, 1) : x
    u = is_vector ? reshape(u, :, 1) : u
    @assert size(x)[2] == size(u)[2]
    tmp = affine_map(nn, x, u)
    _res = T * Flux.logsumexp((1/T)*tmp, dims=1)
    res = is_vector ? reshape(_res, 1) : _res
end

function (nn::PLSE)(x::AbstractArray, u::Convex.AbstractExpr)
    (; T) = nn
    tmp = affine_map(nn, x, u)
    res = [T * Convex.logsumexp((1/T)*tmp)]
end
