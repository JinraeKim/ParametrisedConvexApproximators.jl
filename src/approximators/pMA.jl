struct pMA <: ParametrisedConvexApproximator
    n::Int
    m::Int
    i_max::Int
    NN
end

function pMA(n::Int, m::Int, i_max::Int, h_array::Vector{Int}, act)
    node_array = [n, h_array..., i_max*(m+1)]
    pMA(n, m, i_max, construct_layer_array(node_array, act))
end


function _weight_terms_pMA(x::Vector, u_star_is::Vector)
    # x ∈ ℝ^n -> _weight_terms_pMA(x) ∈ ℝ^(i_max × m)
    hcat((u_star_is |> Map(u_star_i -> u_star_i(x)) |> collect)...)'
end
function _bias_terms_pMA(x::Vector, u_is::Vector, u_star_is::Vector, f::Function)
    # x ∈ ℝ^n -> _bias_terms_pMA(x) ∈ ℝ^(i_max × 1)
    zip(u_is, u_star_is) |> MapSplat((u_i, u_star_i) -> f(x, u_i) - dot(u_star_i(x), u_i)) |> collect |> vcat
end
function _NN_pMA(x, m::Int, i_max::Int, u_is::Vector, u_star_is::Vector, f::Function)
    is_vector = length(size(x)) == 1
    x = is_vector ? reshape(x, :, 1) : x
    d = size(x)[2]
    xs = 1:d |> Map(i -> x[:, i]) |> collect
    _weight_terms = xs |> Map(x -> _weight_terms_pMA(x, u_star_is)) |> collect
    weight_terms = cat(_weight_terms...; dims=3)  # ∈ ℝ^(i_max × m × d)
    _bias_terms = xs |> Map(x -> _bias_terms_pMA(x, u_is, u_star_is, f)) |> collect
    bias_terms = cat(_bias_terms...; dims=3)  # ∈ ℝ^(i_max × 1 × d)
    _nn = reshape(cat(weight_terms, bias_terms; dims=2), i_max*(m+1), d)    # ∈ ℝ^((i_max*(m+1)) × d)
    nn = is_vector ? _nn[:] : _nn
end

"""
Construct pMA by theoretically.
`u_is` denotes a vector consisting of data points u_i ∈ U (i=1, 2, ..., i_max).
`u_star_is` denotes a vector consisting of parametrised subgradients u_star_i ∈ C(ℝⁿ, ℝᵐ).
`f` denotes the true function with two arguments of parameter `x` and decision variable `u`, i.e., f(x, u).
"""
function pMA(n::Int, m::Int, u_is::Vector, u_star_is::Vector, f::Function)
    i_max = length(u_is)
    @assert length(u_star_is) == i_max
    @assert all(length.(u_is) .== m)
    _x = rand(n)  # dummy value
    @assert all(u_star_is |> Map(u_star_i -> length(u_star_i(_x)) == m) |> collect)
    # construction
    pMA(n, m, i_max, x -> _NN_pMA(x, m, i_max, u_is, u_star_is, f))
end

"""
    (nn::pMA)(x, u)

Infer using two input arguments `x` and `u`.
# Notes
size(x) = (n, d)
size(u) = (m, d)
# Types
x and u should be arrays.
For example, u = [1] for the one-element case.
"""
function (nn::pMA)(x::Array, u::Union{Array, Convex.Variable})
    if typeof(u) <: Convex.AbstractExpr
        X = reshape(nn.NN(x), nn.i_max, (nn.m+1))  # size(X1) = (i_max, m)
        tmp = (
               X[:, 1:end-1] * u + (X[:, 1:end-1]*zeros(size(u)) .+ X[:, end:end])
              )  # X1*zeros(size(u)) is for compatibility with Convex.jl
        res = maximum(tmp)
    else
        is_vector = length(size(x)) == 1
        @assert is_vector == (length(size(u)) == 1)
        x = is_vector ? reshape(x, :, 1) : x
        u = is_vector ? reshape(u, :, 1) : u
        if size(x)[2] != size(u)[2]
            error("Different numbers of data")
        end
        d = size(x)[2]
        X = reshape(nn.NN(x), nn.i_max, nn.m+1, d)  # size(X1) = (i_max, m, d)
        tmp = hcat([(X[:, 1:nn.m, i]*u[:, i] .+ X[:, end:end, i]) for i in 1:d]...)
        _res = maximum(tmp, dims=1)
        res = is_vector ? _res[1] : _res
    end
    return res
end
