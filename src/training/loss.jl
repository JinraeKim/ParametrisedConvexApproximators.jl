function SupervisedLearningLoss(normalised_approximator::NormalisedApproximator;
        loss=Flux.Losses.mse,
    )
    @unpack normaliser = normalised_approximator
    return function (d)
        @unpack x, u, f = d
        f = normalise(normaliser, f, :f)
        loss(normalised_approximator(x, u), f)
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
        lim=(nothing, nothing),
    )
    return function (d)
        @unpack x, u, r, x_next = d
        res_optval = zeros(1, size(x)[2])
        Zygote.ignore() do
            res_optval = solve!(normalised_approximator, x_next; lim=lim).optval
        end
        td_target = r + γ*res_optval
        # TODO: appropriate loss...?
        (
         loss(td_target, normalised_approximator(x, u))
         + 1e3 * mean(abs.(normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1))))
         + 1e3 * mean(max.(-normalised_approximator(x, u), zeros(1, size(x)[2])))
        )
    end
end
