abstract type ConvexApproximator <: ParametrisedConvexApproximator end


function _construct_convex_approximator(α_is::AbstractVector, β_is::AbstractVector)
    i_max = length(α_is)
    @assert length(β_is) == i_max
    _α_is = hcat(α_is...)'
    _β_is = hcat(β_is...)'
    i_max, _α_is, _β_is
end

function affine_map(nn::ConvexApproximator, z::AbstractArray)
    (; _α_is, _β_is) = nn
    _α_is * z .+ _β_is
end

function affine_map(nn::ConvexApproximator, z::Convex.AbstractExpr)
    (; _α_is, _β_is) = nn
    _α_is * z + (_α_is*zeros(size(z)) .+ _β_is)
end


include("MA.jl")
include("LSE.jl")
