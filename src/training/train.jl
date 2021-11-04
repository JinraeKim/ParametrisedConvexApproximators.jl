function train_approximator!(approximator, data_train::AbstractDataStructure, data_test::AbstractDataStructure;
        loss=error("Specify loss for your task, e.g., SupervisedLearningLoss(approximator)"),
        opt=ADAM(1e-3),
        epochs=300,
        batchsize=16,
        threshold=1e-5,
        display_period=10,
        # batchsize=7*30,
        # λ=0e-3,
    )
    data_nt_train = Data_to_NamedTuple(data_train)
    data_nt_test = Data_to_NamedTuple(data_test)
    dataloader = Flux.DataLoader(data_nt_train;
                                 batchsize=batchsize,
                                 shuffle=true,
                                 partial=false,
                                )
    # sqnorm(x) = sum(abs2, x)
    # loss_reg(args...) = loss(args...) + λ*sum(sqnorm, Flux.params(approximator))
    for epoch in 0:epochs
        # println("epoch: $(epoch) / $(epochs)")
        if epoch != 0
            _train!(loss, approximator, dataloader, opt)
            # _train!(loss_reg, approximator, dataloader, opt)
        end
        loss_train = loss(data_nt_train)
        loss_test = loss(data_nt_test)
        # display result
        if epoch % display_period == 0
            @show epoch, loss_train, loss_test
        end
        if loss_test < threshold
            println("terminated training loop at epoch = $(epoch); loss_test = $(loss_test) < $(threshold)")
            break
        end
    end
end

function _train!(loss, approximator::AbstractApproximator, data, opt)
    ps = Flux.params(approximator)
    for d in data
        gs = gradient(ps) do 
            training_loss = loss(d)
            training_loss
        end
        Flux.update!(opt, ps, gs)
        # projection for pICNN
        # TODO
    end
    nothing
end
