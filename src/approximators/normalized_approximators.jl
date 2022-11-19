abstract type NormalizedApproximator <: AbstractApproximator end

function (nn::NormalizedApproximator)(network::AbstractApproximator, dataset::DecisionMakingDataset)
    error("Specify the normalization method from dataset")
end


function normalize(nn::NormalizedApproximator, args...; kwargs...)
    error("Define method normalize")
end


function unnormalize(nn::NormalizedApproximator, args...; kwargs...)
    error("Define method unnormalize")
end


struct MaxAbsNormalizedApproximator{T<:AbstractApproximator} <: NormalizedApproximator
    network::T
    condition_max_abs::Union{Array, Nothing}
    decision_max_abs::Union{Array, Nothing}
    cost_max_abs::Union{Array, Nothing}
end

function MaxAbsNormalizedApproximator(
    network::AbstractApproximator,
    dataset::DecisionMakingDataset,
)
    (; conditions, decisions, costs) = dataset
    condition_max_abs = maximum(abs.(hcat(conditions...)), dims=length(size(conditions)))
    decision_max_abs = maximum(abs.(hcat(decisions...)), dims=length(size(decisions)))
    cost_max_abs = maximum(abs.(hcat(costs...)), dims=length(size(costs)))
    MaxAbsNormalizedApproximator(network, condition_max_abs, decision_max_abs, cost_max_abs)
end


function (nn::NormalizedApproximator)(x, u)
    (; network, condition_max_abs, decision_max_abs, cost_max_abs) = nn
    x = normalize(nn, x, :condition)
    u = normalize(nn, u, :decision)
    f = network(x, u)
    f = unnormalize(nn, f, :cost)
    return f
end


function normalize(nn::MaxAbsNormalizedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z = factor != nothing ? z ./ factor : z
    return z
end


function unnormalize(nn::MaxAbsNormalizedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z = factor != nothing ? z .* factor : z
    return z
end


Flux.params(approximator::NormalizedApproximator) = Flux.params(approximator.network)
