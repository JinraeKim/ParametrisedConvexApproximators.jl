# ParametrisedConvexApproximators

[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl) is a Julia package providing predefined pararmetrised convex approximators and related functionalities.

## Installation
To install ParametrisedConvexApproximator,
please open Julia's interactive session (known as REPL) and press `]` key
in the REPL to use the package mode, then type the following command

```julia
pkg> add ParametrisedConvexApproximator
```

### Notes
- Activate multi-threading if available, e.g., `julia -t 7` enabling `7` threads.
It will reduce computation time for minimising networks w.r.t. multiple points.

## Quick Start
ParametrisedConvexApproximators.jl focuses on providing predefined approximators including parametrised convex approximators.
Note that when approximators receive two arguments, the first and second arguments correspond to
condition and decision vectors, usually denoted by `x` and `u`.
### Network construction
```julia
using ParametrisedConvexApproximators
using Flux
using Transducers
using Flux
using Flux: DataLoader
using Random  # for random seed

# construction
Random.seed!(2021)
n, m = 2, 3
i_max = 20
T = 1e-1
h_array = [64, 64]
act = Flux.leakyrelu
plse = PLSE(n, m, i_max, T, h_array, act)  # parametrised log-sum-exp (PLSE) network
x, u = rand(n), rand(m)
f̂ = plse(x, u)
@show f̂
```

```julia
f̂ = [0.3113165298981473]
```

### Network training (as usual in [Flux.jl](https://github.com/FluxML/Flux.jl))
```julia
f(x, u) = [0.5 * ( -(1/length(x))*x'*x + (1/length(u))*u'*u )]  # target function
d = 5_000  # no. of data
# data generation
xs = 1:d |> Map(i -> -ones(n) + 2*ones(n) .* rand(n)) |> collect  # ∈ [-1, 1]^{n}
us = 1:d |> Map(i -> -ones(m) + 2*ones(m) .* rand(m)) |> collect  # ∈ [-1, 1]^{m}
fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
_xs = hcat(xs...)
_us = hcat(us...)
_fs = hcat(fs...)
indices_train, indices_test = partitionTrainTest(1:d, 0.8)  # exported from `ParametrisedConvexApproximators`; train:test = 80:20
dataloader = DataLoader((_xs[:, indices_train], _us[:, indices_train], _fs[:, indices_train],), batchsize=16)
# training
loss(x, u, f) = Flux.Losses.mse(f, plse(x, u))
ps = Flux.params(plse)
opt = ADAM(1e-3)
epochs = 100
for epoch in 0:epochs
    if epoch != 0
        Flux.train!(loss, ps, dataloader, opt)
    end
    if epoch % 10 == 0
        println("epoch = $(epoch) / $(epochs)")
        @show loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test])
    end
end
```

```julia
epoch = 0 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 0.14123486681126265
epoch = 10 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 9.20156070602013e-5
epoch = 20 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 8.428795608023727e-5
epoch = 30 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 6.0756725678295076e-5
epoch = 40 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 7.063355164819796e-5
epoch = 50 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 6.08100029278485e-5
epoch = 60 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 4.319644378100754e-5
epoch = 70 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 7.028416247739685e-5
epoch = 80 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 2.713945900329595e-5
epoch = 90 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 3.524764563503706e-5
epoch = 100 / 100
loss(_xs[:, indices_test], _us[:, indices_test], _fs[:, indices_test]) = 3.102460393375972e-5
```

### Conditional decision making via optimisation (given `x`, find a minimiser `u` and optimal value)
```julia
# optimisation
x = [0.1, 0.2]  # any value
u_min, u_max = -1*ones(m), 1*ones(m)
res = optimise(plse, x; u_min=u_min, u_max=u_max)  # minimsation
@show res  # NamedTuple
```

```julia
res = (minimiser = [-0.027399600684580954, -0.0075144942411888155, -0.015772687025402597], optval = [-0.007673806913150762])
```

## Documentation
### Types
- `AbstractApproximator` is an abstract type of approximator.
- `ParametrisedConvexApproximator <: AbstractApproximator` is an abstract type of parametrised convex approximator.
- `ConvexApproximator <: ParametrisedConvexApproximator` is an abstract type of convex approximator.

### Approximators
- All approximators in ParametrisedConvexApproximators.jl receive two arguments, namely, `x` and `u`.
When `x` and `u` are vectors whose lengths are `n` and `m`, respectively,
the output of an approximator is **one-length vector**.
    - Note that `x` and `u` can be matrices, whose sizes are `(n, d)` and `(m, d)`,
    for evaluations of `d` pairs of `x`'s and `u`'s.
    In this case, the output's size is `(1, d)`.

- The list of predefined approximators
    - `FNN::AbstractApproximator`: feedforward neural network
    - `MA::ConvexApproximator`: max-affine (MA) network [1]
    - `LSE::ConvexApproximator`: log-sum-exp (LSE) network [1]
    - `PICNN::ParametrisedConvexApproximator`: partially input-convex neural network (PICNN) [2]
    - `PMA::ParametrisedConvexApproximator`: parametrised MA (PMA) network [3]
    - `PLSE::ParametrisedConvexApproximator`: parametrised LSE (PLSE) network [3]

### Utilities
- `(nn::approximator)(x, u)` gives an inference (approximate function value).
- `res = optimise(approximator, x; u_min=nothing, u_max=nothing)` provides
minimiser and optimal value (optval) for given `x` as `res.minimiser` and `res.optval`
considering box constraints of `u >= u_min` and `u <= u_max` (element-wise).
    - The condition variable `x` can be a vector, i.e., `size(x) = (n,)`,
    or a matrix for parallel solve (via multi-threading), i.e., `size(x) = (n, d)`.


## Notes
### To-do list
- [ ] Q-learning via differentiable convex programming
- [ ] Greatest convex minorant (GCM) approach


### Benchmark
- Note: to avoid first-run latency due to JIT compilation of Julia, the elapsed times are obtained from second-run.
The following result is from `test/basic.jl`.
- Note: run on M1 Macbook Air (Apple silicon).
- `n`: dimension of condition variable `x`
- `m`: dimension of decision variable `u`
- `epochs`: training epochs
- `approximator`: the type of approximator
- `minimisers_diff_norm_mean`: the mean value of 2-norm of the difference between true and estimated minimisers
- `optvals_diff_abs_mean`: the mean value of absolute of the difference between true and estimated optimal values
- `no_of_minimiser_success_cases`: failure means no minimiser has been found (`NaN`)
- `no_of_optval_success_cases`: failure means invalid optimal value has been found (`-Inf` or `Inf)
- `number_of_parameters`: the number of network parameters

### TODO: add the latest simulation result
```julia
```

## References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
- [2] B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, Jul. 2017, pp. 146–155.
- [3] J. Kim and Y. Kim, “Parametrised Convex Universa Approximators,” IEEE Transactions on Neural Networks and Learning Systems, In preparation.
