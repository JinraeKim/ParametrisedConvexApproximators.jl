function train_approximator!(approximator, data_train, data_test)
    dataloader = Flux.DataLoader(data_train; batchsize=64, shuffle=true)
    epochs = 100
    opt = ADAM(1e-3)
    loss(x, u, f) = Flux.Losses.mse(approximator(x, u), f)
    for epoch in 0:epochs
        if epoch != 0
            _train!(loss, approximator, dataloader, opt)
        end
        # display result
        if epoch % 10 == 0
            @show epoch, loss(data_train...), loss(data_test...)
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
