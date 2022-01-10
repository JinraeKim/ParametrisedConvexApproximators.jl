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
The following result is from `main/basic.jl`.
- Note: run on ADM Ryzen:tm: 9 5900X.
- Note: the result may be slightly different from the original paper [3].

- `n`: dimension of condition variable `x`
- `m`: dimension of decision variable `u`
- `epochs`: training epochs
- `approximator`: the type of approximator
- `minimisers_diff_norm_mean`: the mean value of 2-norm of the difference between true and estimated minimisers
- `optvals_diff_abs_mean`: the mean value of absolute of the difference between true and estimated optimal values
- `no_of_minimiser_success_cases`: failure means no minimiser has been found (`NaN`)
- `no_of_optval_success_cases`: failure means invalid optimal value has been found (`-Inf` or `Inf)
- `number_of_parameters`: the number of network parameters


#### Results
- Run as
```julia
include("test/basic.jl")
main(1, 1); main(61, 20); main(376, 17)  # second run
```

- (n, m) = (1, 1)
```julia
 Row │ n      m      epochs  approximator  optimise_time_mean  minimisers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimiser_success  no_of_optval_success  number_of_parameters
     │ Int64  Int64  Int64   String        Float64             Float64                    Float64                Int64                    Int64                 Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1      1     100  FNN                  0.000777166                 0.0339567              0.00835098                      500                   500                  4417
   2 │     1      1     100  MA                   0.000461839                 0.0517864              0.131502                        500                   500                    90
   3 │     1      1     100  LSE                  0.0389891                   0.00626931             0.126809                        500                   500                    90
   4 │     1      1     100  PICNN                0.0108027                   0.0512426              0.100103                        500                   498                 25608
   5 │     1      1     100  PMA                  0.00109627                  0.320571               0.0189876                       500                   500                  8188
   6 │     1      1     100  PLSE                 0.00647962                  0.00967128             0.00171072                      500                   500                  8188
```

- (n, m) = (61, 20)
```julia
 Row │ n      m      epochs  approximator  optimise_time_mean  minimisers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimiser_success  no_of_optval_success  number_of_parameters
     │ Int64  Int64  Int64   String        Float64             Float64                    Float64                Int64                    Int64                 Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │    61     20     100  FNN                   0.0354606                     3.19316              0.172008                       500                   500                  9473
   2 │    61     20     100  MA                    0.00959303                    4.45146              0.111078                       500                   500                  2460
   3 │    61     20     100  LSE                   0.0880122                     2.71899              0.071351                       500                   500                  2460
   4 │    61     20     100  PICNN                 0.12355                       3.573                0.0946406                      500                   500                 39507
   5 │    61     20     100  PMA                   0.0185542                     3.40139              0.0563801                      500                   500                 49078
   6 │    61     20     100  PLSE                  0.0111066                     1.97364              0.089203                       500                   500                 49078
```

- (n, m) = (376, 17)
```julia
 Row │ n      m      epochs  approximator  optimise_time_mean  minimisers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimiser_success  no_of_optval_success  number_of_parameters
     │ Int64  Int64  Int64   String        Float64             Float64                    Float64                Int64                    Int64                 Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │   376     17     100  FNN                   0.0312941                     2.95216              0.165991                       500                   500                 29441
   2 │   376     17     100  MA                    0.0110139                     4.09744              0.125646                       500                   500                 11820
   3 │   376     17     100  LSE                   0.243551                      4.01312              0.115364                       500                   500                 11820
   4 │   376     17     100  PICNN                 0.0736546                     3.78249              0.131185                       500                   500                 84534
   5 │   376     17     100  PMA                   0.0162996                     3.51627              0.0878748                      500                   500                 63388
   6 │   376     17     100  PLSE                  0.00704432                    0.5204               0.0815478                      500                   500                 63388
```


## References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
- [2] [B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, Jul. 2017, pp. 146–155.](http://proceedings.mlr.press/v70/amos17b.html)
- [3] J. Kim and Y. Kim, “Parametrized Convex Universal Approximators for Decision Making Problems,” IEEE Transactions on Neural Networks and Learning Systems, In preparation.
