"""
Max-affine neural network [1].

# Note
If you specify `n` or `m`, it can also be regarded as bivariate function.

# Variables
x ∈ ℝ^n
u ∈ ℝ^m
z = [xᵀ, uᵀ]ᵀ ∈ ℝ^(n+m)  # l = n+m
α_is: a vector of subgradients, i.e., α_is[i] ∈ ℝ^(n+m).
β_is: a vector of bias terms, i.e., β_is[i] ∈ ℝ.

# References
[1] G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.
"""
struct MA <: ConvexApproximator
    l::Int  # n+m
    n::Int  # the first variable for bivariate function
    m::Int  # the second variable for bivariate function
    i_max::Int
    _α_is::Matrix
    _β_is::Matrix
    function MA(α_is::Vector, β_is::Vector; n=nothing, m=nothing)
        l, i_max, _α_is, _β_is = _construct_convex_approximator(α_is, β_is)
        if n != nothing && m != nothing
            @assert l == n + m
        elseif n != nothing
            m = l - n
            @assert m > 0
        elseif m != nothing
            n = l - m
            @assert n > 0
        end
        new(l, n, m, i_max, _α_is, _β_is)
    end
end
Flux.@functor MA (_α_is, _β_is,)

function (nn::MA)(z::Array)
    is_vector = length(size(z)) == 1
    z_affine = affine_map(nn, z)
    _res = maximum(z_affine; dims=1)
    res = is_vector ? reshape(_res, 1) : _res
end

function (nn::MA)(z::Convex.AbstractExpr)
    z_affine = affine_map(nn, z)
    _res = [maximum(z_affine)]
end

"""
Considering bivariate function approximator
"""
function (nn::MA)(x::Array, u::Union{Array, Convex.AbstractExpr})
    nn(vcat(x, u))
end
