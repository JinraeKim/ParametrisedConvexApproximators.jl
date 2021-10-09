function SupervisedLearningLoss(normalised_approximator::NormalisedApproximator;
        loss=Flux.Losses.mse,
    )
    @unpack normaliser = normalised_approximator
    return function (x, u, f)
        f_normal = normalise(normaliser, f, :f)
        loss(normalised_approximator(x, u; output_normalisation=true), f_normal)
    end
end

"""
x: state
u: input (action)
r: reward
x_next: next state
"""
function QLearningLoss(normalised_approximator::NormalisedApproximator;
        loss=Flux.Losses.mse,
        γ=1.0,
    )
    return function (x, u, r, x_next)
        res = solve!(normalised_approximator, x_next)
        td_target = r + γ*res.optval
        loss(td_target, normalised_approximator(x, u))
    end
end
