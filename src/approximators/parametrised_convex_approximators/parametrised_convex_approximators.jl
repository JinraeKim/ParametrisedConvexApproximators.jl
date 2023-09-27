abstract type ParametrisedConvexApproximator <: AbstractApproximator end


function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::AbstractArray)
    (; NN, i_max, m) = nn
    d = size(x)[2]
    # Modified PLSE
    X = reshape(NN(x), i_max, m+1, d)
    if typeof(nn) == PLSE
        if nn.strict
            # to avoid mutation for Zygote
            # Y =
            # 0 0 0 0 0 0 ... 0   b0
            # a11 a12 a13 ... a1m b1
            # a21 a22 a23 ... a2m b2
            # ...
            # aI1 aI2 aI3 ... aIm bI
            dummy = zeros(1, m, d)
            dummy = hcat(dummy, ones(1, 1, d))
            dummy = vcat(dummy, ones(i_max-1, m+1, d))
            Y = X .* dummy
        else
            Y = X
        end
    else
        Y = X
    end
    tmp = hcat([(Y[:, 1:m, i]*u[:, i] .+ Y[:, m+1, i]) for i in 1:d]...)
end


function _affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray)
    (; NN, i_max, m) = nn
    # Modified PLSE
    X = reshape(NN(x), i_max, m+1)
    if typeof(nn) == PLSE
        if nn.strict
            # Y =
            # 0 0 0 0 0 0 ... 0   b0
            # a11 a12 a13 ... a1m b1
            # a21 a22 a23 ... a2m b2
            # ...
            # aI1 aI2 aI3 ... aIm bI
            dummy = zeros(1, m)
            dummy = hcat(dummy, ones(1, 1))
            dummy = vcat(dummy, ones(i_max-1, m+1))
            Y = X .* dummy
        else
            Y = X
        end
    else
        Y = X
    end
    return Y
    # tmp = (
    #        Y[:, 1:m] * u + (Y[:, 1:m]*zeros(size(u)) .+ Y[:, m+1])
    #       )  # X1*zeros(size(u)) is for compatibility with Convex.jl
end


function affine_map(nn::ParametrisedConvexApproximator, x::AbstractArray, u::Convex.AbstractExpr)
    θ = _affine_map(nn, x)
    A = θ[:, 1:end-1]
    B = θ[:, end]
    tmp = A*u + B
    return tmp 
end


include("PMA.jl")
include("PLSE.jl")
include("PICNN.jl")
include("convex_approximators/convex_approximators.jl")
