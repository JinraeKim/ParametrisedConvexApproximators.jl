Flux.params(nn::fNN) = Flux.params(nn.NN...)
Flux.params(nn::pMA) = Flux.params(nn.NN...)
