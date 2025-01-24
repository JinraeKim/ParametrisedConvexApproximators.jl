"""
Extended parameterized log-sum-exp (EPLSE) network.
Use `PLSEPlus` if you want to use PLSE+ network instead of PLSE network as the parameterized convex minorant [1].

# Refs
[1] J. Kim and Y. Kim, "Parameterized Convex Minorant for Objective Function Approximation in Amortized Optimization", in preparation.
"""
struct EPLSE <: AbstractApproximator
    plse::PLSE
    nn
    min_decision
    max_decision
end
Flux.@layer EPLSE trainable=(plse, nn,)


function (network::EPLSE)(x, u; initial_guess=nothing,)
    (; plse, nn, min_decision, max_decision) = network
    u_star = minimise(network, x; min_decision, max_decision, initial_guess,)
    plse(x, u) + max.(nn(x, u) - nn(x, u_star), 0)
end

