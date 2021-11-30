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
    # node_array1 = [n, h_array..., i_max*m]
    # node_array2 = [n, h_array..., i_max*1]
    # PMA(n, m, i_max, construct_layer_array(node_array1, act), construct_layer_array(node_array2, act))
end


# function _weight_terms_PMA(x::Vector, u_star_is::Vector)
#     # x ∈ ℝ^n -> _weight_terms_PMA(x) ∈ ℝ^(i_max × m)
#     hcat((u_star_is |> Map(u_star_i -> u_star_i(x)) |> collect)...)'
# end
# function _bias_terms_PMA(x::Vector, u_is::Vector, u_star_is::Vector, f::Function)
#     # x ∈ ℝ^n -> _bias_terms_PMA(x) ∈ ℝ^(i_max × 1)
#     zip(u_is, u_star_is) |> MapSplat((u_i, u_star_i) -> f(x, u_i) - dot(u_star_i(x), u_i)) |> collect |> vcat
# end

# """
# auxiliary struct
# """
# struct _NN_PMA
#     m::Int
#     i_max::Int
#     u_is::Vector
#     u_star_is::Vector
#     f::Function
# end
# # u_star_is may be a vector of functions, so it would be neglected when training via Flux interface.
# Flux.@functor _NN_PMA (u_is, u_star_is,)

# function (nn::_NN_PMA)(x)
#     @unpack m, i_max, u_is, u_star_is, f = nn
#     is_vector = length(size(x)) == 1
#     x = is_vector ? reshape(x, :, 1) : x
#     d = size(x)[2]
#     xs = 1:d |> Map(i -> x[:, i]) |> collect
#     _weight_terms = xs |> Map(x -> _weight_terms_PMA(x, u_star_is)) |> collect
#     weight_terms = cat(_weight_terms...; dims=3)  # ∈ ℝ^(i_max × m × d)
#     _bias_terms = xs |> Map(x -> _bias_terms_PMA(x, u_is, u_star_is, f)) |> collect
#     bias_terms = cat(_bias_terms...; dims=3)  # ∈ ℝ^(i_max × 1 × d)
#     _nn = reshape(cat(weight_terms, bias_terms; dims=2), i_max*(m+1), d)    # ∈ ℝ^((i_max*(m+1)) × d)
#     nn = is_vector ? _nn[:] : _nn
# end

# """
# Construct PMA theoretically.
# `u_is` denotes a vector consisting of data points u_i ∈ U (i=1, 2, ..., i_max).
# `u_star_is` denotes a vector consisting of parametrised subgradients u_star_i ∈ C(ℝⁿ, ℝᵐ).
# `f` denotes the true function with two arguments of parameter `x` and decision variable `u`, i.e., f(x, u).
# """
# function PMA(n::Int, m::Int, u_is::Vector, u_star_is::Vector, f::Function)
#     error("Theoretical construction of PMA has been deprecated")
#     i_max = length(u_is)
#     @assert length(u_star_is) == i_max
#     @assert all(length.(u_is) .== m)
#     _x = rand(n)  # dummy value
#     @assert all(u_star_is |> Map(u_star_i -> length(u_star_i(_x)) == m) |> collect)
#     # construction
#     _nn_pma = Chain(_NN_PMA(m, i_max, u_is, u_star_is, f))
#     PMA(n, m, i_max, _nn_pma)
# end

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
