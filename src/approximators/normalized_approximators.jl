abstract type NormalisedApproximator <: AbstractApproximator end

function (nn::NormalisedApproximator)(network::AbstractApproximator, dataset::DecisionMakingDataset)
    error("Specify the normalisation method from dataset")
end


function normalise(nn::NormalisedApproximator, args...; kwargs...)
    error("Define method normalise")
end


function unnormalise(nn::NormalisedApproximator, args...; kwargs...)
    error("Define method unnormalise")
end


struct MaxAbsNormalisedApproximator{T<:AbstractApproximator} <: NormalisedApproximator
    network::T
    condition_max_abs::Union{Array, Nothing}
    decision_max_abs::Union{Array, Nothing}
    cost_max_abs::Union{Array, Nothing}
end
Flux.@functor MaxAbsNormalisedApproximator (network,)

function MaxAbsNormalisedApproximator(
    network::AbstractApproximator,
    dataset::DecisionMakingDataset,
)
    (; conditions, decisions, costs) = dataset
    condition_max_abs = maximum(abs.(hcat(conditions...)), dims=length(size(conditions)))
    decision_max_abs = maximum(abs.(hcat(decisions...)), dims=length(size(decisions)))
    cost_max_abs = maximum(abs.(hcat(costs...)), dims=length(size(costs)))
    MaxAbsNormalisedApproximator(network, condition_max_abs, decision_max_abs, cost_max_abs)
end


function (nn::NormalisedApproximator)(x, u)
    (; network, condition_max_abs, decision_max_abs, cost_max_abs) = nn
    x = normalise(nn, x, :condition)
    u = normalise(nn, u, :decision)
    f = network(x, u)
    f = unnormalise(nn, f, :cost)
    return f
end


function normalise(nn::MaxAbsNormalisedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z = factor != nothing ? z ./ factor : z
    return z
end


function unnormalise(nn::MaxAbsNormalisedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z = factor != nothing ? z .* factor : z
    return z
end
