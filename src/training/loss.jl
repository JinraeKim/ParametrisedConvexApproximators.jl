function SupervisedLearningLoss(normalised_approximator::NormalisedApproximator;
        loss=Flux.Losses.mse,
    )
    @unpack normaliser = normalised_approximator
    return function (d)
        @unpack x, u, f = d
        # f_normal = normalise(normaliser, f, :f)
        # loss(normalised_approximator(x, u; output_normalisation=true), f_normal)
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
        # loss=Flux.Losses.huber_loss,
        γ=1.00,
        lim=(nothing, nothing),
        verbose=false
    )
    @unpack normaliser = normalised_approximator
    return function (data)
        @unpack x, u, r, x_next = data
        d = size(x)[2]
        res_optval = zeros(1, size(x)[2])
        Zygote.ignore() do
            res_optval = solve!(normalised_approximator, x_next; lim=lim).optval
        end
        td_target = r + γ*res_optval
        # td_target = normalise(normaliser, r + γ*res_optval, :f)
        # TODO: decide an appropriate loss...?
        loss_plain = loss(td_target, normalised_approximator(x, u))
        # loss_plain = loss(td_target, normalised_approximator(x, u; output_normalisation=true))
        # loss(td_target, normalised_approximator(x, u; output_normalisation=true))
        # loss_origin_penalty = 1e4 * Flux.Losses.huber_loss(normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1)), zeros(1, 1))
        # loss_origin_penalty = 0.0
        loss_origin_penalty = 1e1 * Flux.Losses.mse(normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1)), zeros(1, 1))
        # loss_origin_penalty = 1e4 * Flux.Losses.mse(
        #                                             normalise(normaliser, normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1)), :f),
        #                                             normalise(normaliser, zeros(1, 1), :f)
        #                                            )
        # loss_origin_penalty = 1e4 * mean(abs.(normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1))))
        # + 1e4 * Flux.Losses.huber_loss(normalised_approximator(zeros(size(x)[1], 1), zeros(size(u)[1], 1); output_normalisation=true), zeros(1, 1))
        # loss_positive_definiteness_penalty = 1e4 * mean(softplus.(-normalised_approximator(x, u)))
        # loss_positive_definiteness_penalty = 0.0
        loss_positive_definiteness_penalty = 1e1 * mean(max.(-normalised_approximator(x, u), zeros(1, size(x)[2])))
        # loss_positive_definiteness_penalty = 1e4 * mean(max.(
        #                                                      normalise(normaliser, -normalised_approximator(x, u), :f),
        #                                                      normalise(normaliser, zeros(1, size(x)[2]), :f)
        #                                                     ))
        # + 1e4 * mean(softplus.(-normalised_approximator(x, u; output_normalisation=true)))
        loss_total = (
                      loss_plain
                      + loss_origin_penalty
                      + loss_positive_definiteness_penalty
                     )
        if verbose
            @show loss_total, loss_plain , loss_origin_penalty, loss_positive_definiteness_penalty
        end
        loss_total
    end
end
