abstract type NormalizedApproximator <: AbstractApproximator end


function normalize(nn::NormalizedApproximator, args...; kwargs...)
    error("Define method normalize")
end


function unnormalize(nn::NormalizedApproximator, args...; kwargs...)
    error("Define method unnormalize")
end


struct MaxAbsNormalizedApproximator <: NormalizedApproximator
    network::AbstractApproximator
    condition_max_abs::Union{Array, Nothing}
    decision_max_abs::Union{Array, Nothing}
    cost_max_abs::Union{Array, Nothing}
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
