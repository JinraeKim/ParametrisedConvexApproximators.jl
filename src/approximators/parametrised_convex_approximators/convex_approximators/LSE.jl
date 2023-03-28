"""
Log-sum-exp neural network [1].

# Note
If you specify `n` or `m`, it can also be regarded as bivariate function.

# Variables
x ∈ ℝ^n
u ∈ ℝ^m
z = [xᵀ, uᵀ]ᵀ ∈ ℝ^(n+m)  # l = n+m
α_is: a vector of subgradients, i.e., α_is[i] ∈ ℝ^(n+m).
β_is: a vector of bias terms, i.e., β_is[i] ∈ ℝ.
T > 0: temperature

# References
[1] G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.
"""
struct LSE <: ConvexApproximator
    n::Int  # the first variable for bivariate function
    m::Int  # the second variable for bivariate function
    i_max::Int
    T::Real
    _α_is
    _β_is
end
Flux.@functor LSE (_α_is, _β_is,)
function LSE(n::Int, m::Int, i_max::Int, T::Real)
    @assert T > 0
    α_is = [Flux.glorot_uniform(n+m) for i in 1:i_max]
    β_is = [Flux.glorot_uniform(1) for i in 1:i_max]
    i_max, _α_is, _β_is = _construct_convex_approximator(α_is, β_is)
    LSE(n, m, i_max, T, _α_is, _β_is)
end



"""
Considering univariate function approximator
"""
function (nn::LSE)(z::AbstractArray)
    is_vector = length(size(z)) == 1
    (; T) = nn
    z_affine = affine_map(nn, z)
    _res = T * Flux.logsumexp((1/T)*z_affine; dims=1)
    res = is_vector ? reshape(_res, 1) : _res
end

function (nn::LSE)(z::Convex.AbstractExpr)
    (; T) = nn
    z_affine = affine_map(nn, z)
    _res = [T * Convex.logsumexp((1/T)*z_affine)]
end

"""
Considering bivariate function approximator
"""
function (nn::LSE)(x::AbstractArray, u::AbstractArray)
    nn(vcat(x, u))
end

function (nn::LSE)(x::AbstractArray, u::Convex.AbstractExpr)
    nn(vcat(Convex.Constant(x), u))  # if x is a ComponentArray, vcat(⋅, ⋅) becomes ambiguous
end
