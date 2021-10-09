function SupervisedLearningLoss(normalised_approximator::NormalisedApproximator; loss=Flux.Losses.mse)
    @unpack normaliser = normalised_approximator
    return function (x, u, f)
        f_normal = normalise(normaliser, f, :f)
        loss(normalised_approximator(x, u; output_normalisation=true), f_normal)
    end
end

# TODO: make it normalised version?
# """
# x: state
# u: input (action)
# r: reward
# x_next: next state
# """
# function QLearningLoss(approx; loss=Flux.Losses.mse, γ=1.0)
#     return function (x, u, r, x_next)
#         # res = solve!(approx, x_next, u)
#         @unpack optval = res
#         td_target = r + γ*optval
#         loss(td_target, approx(x, u))
#     end
# end
