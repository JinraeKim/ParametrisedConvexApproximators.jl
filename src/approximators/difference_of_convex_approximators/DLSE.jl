"""
    DLSE(NN1::LSE, NN2::LSE)

Difference of log-sum-exp (DLSE) network.

# Refs
[1] G. C. Calafiore, S. Gaubert, and C. Possieri, “A Universal Approximation Result for Difference of Log-Sum-Exp Neural Networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5603–5612, Dec. 2020, doi: 10.1109/TNNLS.2020.2975051.
"""
struct DLSE <: DifferenceOfConvexApproximator
    NN1::LSE
    NN2::LSE
end


function (nn::DLSE)(x::AbstractArray, u::AbstractArray)
    f1 = nn.NN1(x, u)
    f2 = nn.NN2(x, u)
    return f1 - f2
end


Flux.params(approximator::DLSE) = Flux.params([
                                               Flux.params(approximator.NN1)...,
                                               Flux.params(approximator.NN2)...,
                                              ])
