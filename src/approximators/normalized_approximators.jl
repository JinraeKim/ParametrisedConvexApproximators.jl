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
    c = hcat(conditions...)
    d = hcat(decisions...)
    J = hcat(costs...)
    condition_max_abs = maximum(abs.(c), dims=length(size(c)))
    decision_max_abs = maximum(abs.(d), dims=length(size(d)))
    cost_max_abs = maximum(abs.(J), dims=length(size(J)))
    MaxAbsNormalisedApproximator(network, condition_max_abs, decision_max_abs, cost_max_abs)
end


function (nn::NormalisedApproximator)(x, u)
    (; network, condition_max_abs, decision_max_abs, cost_max_abs) = nn
    x_new = normalise(nn, x, :condition)
    u_new = normalise(nn, u, :decision)
    f = network(x_new, u_new)
    f_new = unnormalise(nn, f, :cost)
    return f_new
end


function normalise(nn::MaxAbsNormalisedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z_new = factor != nothing ? z ./ factor : z
    return z_new
end


function unnormalise(nn::MaxAbsNormalisedApproximator, z, which::Symbol)
    @assert which in (:condition, :decision, :cost)
    factor = getproperty(nn, Symbol(String(which) * "_max_abs"))
    z_new = factor != nothing ? z .* factor : z
    return z_new
end
