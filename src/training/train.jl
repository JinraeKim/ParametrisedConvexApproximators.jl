function train_approximator!(approximator, xuf_data_train::xufData, xuf_data_test::xufData;
        loss=SupervisedLearningLoss(approximator; loss=Flux.Losses.mse),
        # loss=error("Specify loss for your task, e.g., SupervisedLearningLoss(approximator)"),
        opt=ADAM(1e-3),
        epochs=300,
        batchsize=16,
        # λ=0e-3,
    )
    xuf_nt_train = Data_to_NamedTuple(xuf_data_train)
    xuf_nt_test = Data_to_NamedTuple(xuf_data_test)
    dataloader = Flux.DataLoader(xuf_nt_train;
                                 batchsize=batchsize, shuffle=true)
    # sqnorm(x) = sum(abs2, x)
    # loss_reg(args...) = loss(args...) + λ*sum(sqnorm, Flux.params(approximator))
    for epoch in 0:epochs
        if epoch != 0
            _train!(loss, approximator, dataloader, opt)
            # _train!(loss_reg, approximator, dataloader, opt)
        end
        # display result
        if epoch % 10 == 0
            @show epoch, loss(xuf_nt_train...), loss(xuf_nt_test...)
            # @show epoch, loss(xuf_nt_train...), loss_reg(xuf_nt_train...), loss(xuf_nt_test...)
        end
    end
end

function _train!(loss, approximator::AbstractApproximator, data, opt)
    ps = Flux.params(approximator)
    for d in data
        gs = gradient(ps) do 
            training_loss = loss(d.x, d.u, d.f)
            training_loss
        end
        Flux.update!(opt, ps, gs)
        # projection for pICNN
        # TODO
    end
    nothing
end
