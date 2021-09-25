"""
Max-affine neural networks [1].

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
    i_max::Int
    _α_is::Matrix
    _β_is::Matrix
    function MA(α_is::Vector, β_is::Vector)
        l = length(α_is[1])
        i_max = length(α_is)
        @assert length(β_is) == i_max
        _α_is = hcat(α_is...)'
        _β_is = hcat(β_is...)'
        new(l, i_max, _α_is, _β_is)
    end
end
Flux.@functor MA (_α_is, _β_is,)

function (nn::MA)(z::Array)
    is_vector = length(size(z)) == 1
    @unpack _α_is, _β_is = nn
    _res = maximum(_α_is * z .+ _β_is; dims=1)
    res = is_vector ? reshape(_res, 1) : _res
end

function (nn::MA)(z::Convex.AbstractExpr)
    @unpack _α_is, _β_is = nn
    _res = maximum(_α_is * z + (_α_is*zeros(size(z)) .+ _β_is))
end

"""
Considering two-variable cases
"""
function (nn::MA)(x::Array, u::Union{Array, Convex.AbstractExpr})
    nn(vcat(x, u))
end
