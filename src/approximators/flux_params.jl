Flux.params(nn::FNN) = Flux.params(nn.NN...)
Flux.params(nn::PMA) = Flux.params(nn.NN...)
Flux.params(nn::PLSE) = Flux.params(nn.NN...)
Flux.params(normalised_approximator::NormalisedApproximator) = Flux.params(normalised_approximator.approximator)
