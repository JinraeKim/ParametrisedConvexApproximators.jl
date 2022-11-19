abstract type ParametrisedConvexApproximator <: AbstractApproximator end


function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::AbstractArray)
    (; NN, i_max, m) = nn
    d = size(x)[2]
    X = reshape(NN(x), i_max, m+1, d)
    # Modified PLSE
    if typeof(nn) == PLSE
        if nn.strict
            # to avoid mutation for Zygote
            dummy = zeros(1, m, d)
            dummy = hcat(dummy, ones(1, 1, d))
            dummy = vcat(dummy, ones(i_max-1, m+1, d))
            X = X .* dummy
        end
    end
    tmp = hcat([(X[:, 1:m, i]*u[:, i] .+ X[:, m+1, i]) for i in 1:d]...)
end


function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::Convex.AbstractExpr)
    (; NN, i_max, m) = nn
    X = reshape(NN(x), i_max, m+1)  # size(X1) = (i_max, m)
    # Modified PLSE
    if typeof(nn) == PLSE
        if nn.strict
            dummy = zeros(1, m)
            dummy = hcat(dummy, ones(1, 1))
            dummy = vcat(dummy, ones(i_max-1, m+1))
            X = X .* dummy
        end
    end
    tmp = (
           X[:, 1:m] * u + (X[:, 1:m]*zeros(size(u)) .+ X[:, m+1])
          )  # X1*zeros(size(u)) is for compatibility with Convex.jl
end


include("PMA.jl")
include("PLSE.jl")
include("PICNN.jl")
include("convex_approximators/convex_approximators.jl")
