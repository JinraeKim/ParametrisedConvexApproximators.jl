# Notes
- The name is changed from `ParametrisedConvexApproximators` to `ParameterizedConvexApproximators`.


# ParameterizedConvexApproximators
[ParameterizedConvexApproximators.jl](https://github.com/JinraeKim/ParameterizedConvexApproximators.jl)
is a Julia package providing predefined parameterized convex approximators and related functionalities.
An official package for simulation in [3].


## Installation
To install ParameterizedConvexApproximator,
please open Julia's interactive session (a.k.a REPL) and press `]` key
in the REPL to use the package mode, then type the following command

```julia
pkg> add ParameterizedConvexApproximator
```

### Notes
- Activate multi-threading if available, e.g., `julia -t 7` enabling `7` threads.
It will reduce computation time to obtain multiple minimizers.

## Quick Start
ParameterizedConvexApproximators.jl focuses on providing predefined approximators
including parameterized convex approximators.
Note that when approximators receive two arguments, the first and second arguments correspond to
condition and decision vectors, usually denoted by `x` and `u`.

### Network construction
```julia
using ParameterizedConvexApproximators
using Flux
using Random  # for random seed

# construction
seed = 2022
Random.seed!(seed)
n, m = 3, 2
i_max = 20
T = 1.0
h_array = [64, 64]
act = Flux.leakyrelu
network = PLSE(n, m, i_max, T, h_array, act)  # parameterized log-sum-exp (PLSE) network
x, u = rand(n), rand(m)
f̂ = network(x, u)
@show f̂
```

```julia
f̂ = [2.9972948397933683]  # size(f̂) = (1,)
```

### Prepare dataset
```julia
min_condition = -ones(n)
max_condition = +ones(n)
min_decision = -ones(m)
max_decision = +ones(m)
func_name = :quadratic  # f(x, u) = transpose(x)*x + transpose(u)*u
N = 5_000

dataset = SimpleDataset(
    func_name;
    N=N, n=n, m=m, seed=seed,
    min_condition=min_condition,
    max_condition=max_condition,
    min_decision=min_decision,
    max_decision=max_decision,
)
```

### Network training
```julia
epochs = 200
trainer = SupervisedLearningTrainer(dataset, network; optimizer=Adam(1e-4))

@show get_loss(trainer, :train)
@show get_loss(trainer, :validate)
for epoch in 1:epochs
    println("epoch: $(epoch)/$(epochs)")
    Flux.train!(trainer)
end
@show get_loss(trainer, :test)
```

```julia
get_loss(trainer, :train) = 2.2485971763998576
get_loss(trainer, :validate) = 2.288581633859485

...


epoch: 199/200
loss_train: 0.0001672882024953069
loss_validate: 0.0002682474510180785
epoch: 200/200
loss_train: 0.00016627992642691757
loss_validate: 0.0002670633060886428

get_loss(trainer, :test) = 0.00024842624962788054
```

### Conditional decision making via optimization (given `x`, find a minimizer `u` and optimal value)
```julia
# optimization
Random.seed!(seed)
x = rand(n)  # any value
res = optimize(network, x; u_min=min_decision, u_max=max_decision)  # minimsation
@show res  # NamedTuple
@show dataset[:train].metadata.target_function(x, res.minimizer)
```

```julia
res = (minimizer = [-0.006072644282314285, 0.009363546949627488], optval = [1.1356929322723475])
(dataset[:train]).metadata.target_function(x, res.minimizer) = 1.1025790963107207
```

## Documentation
### Types
- `AbstractApproximator` is an abstract type of approximator.
- `ParameterizedConvexApproximator <: AbstractApproximator` is an abstract type of parameterized convex approximator.
- `ConvexApproximator <: ParameterizedConvexApproximator` is an abstract type of convex approximator.

### Approximators
- All approximators in ParameterizedConvexApproximators.jl receive two arguments, namely, `x` and `u`.
When `x` and `u` are vectors whose lengths are `n` and `m`, respectively,
the output of an approximator is **one-length vector**.
    - Note that `x` and `u` can be matrices, whose sizes are `(n, d)` and `(m, d)`,
    for evaluations of `d` pairs of `x`'s and `u`'s.
    In this case, the output's size is `(1, d)`.

- The list of predefined approximators
    - `FNN::AbstractApproximator`: feedforward neural network
    - `MA::ConvexApproximator`: max-affine (MA) network [1]
    - `LSE::ConvexApproximator`: log-sum-exp (LSE) network [1]
    - `PICNN::ParameterizedConvexApproximator`: partially input-convex neural network (PICNN) [2]
    - `PMA::ParameterizedConvexApproximator`: parameterized MA (PMA) network [3]
    - `PLSE::ParameterizedConvexApproximator`: parameterized LSE (PLSE) network [3]

### Utilities
- `(nn::approximator)(x, u)` gives an inference (approximate function value).
- `res = optimize(approximator, x; u_min=nothing, u_max=nothing)` provides
minimizer and optimal value (optval) for given `x` as `res.minimizer` and `res.optval`
considering box constraints of `u >= u_min` and `u <= u_max` (element-wise).
    - The condition variable `x` can be a vector, i.e., `size(x) = (n,)`,
    or a matrix for parallel solve (via multi-threading), i.e., `size(x) = (n, d)`.

### Dataset
- `SimpleDataset <: DecisionMakingDataset` is used for analytically-expressed cost functions.

### Trainer
- `SupervisedLearningTrainer`


## Benchmark
The benchmark result is reported in [3] using ParameterizedConvexApproximator.jl v0.1.1.
The following benchmark result may be slightly different from the original paper.
- Note: to avoid first-run latency due to JIT compilation of Julia, the elapsed times are obtained from second-run.
The following result is from `main/basic.jl` in ParameterizedConvexApproximators.jl v0.1.1 (currently deprecated).
- Note: it was run on ADM Ryzen:tm: 9 5900X.

- `n`: dimension of condition variable `x`
- `m`: dimension of decision variable `u`
- `epochs`: training epochs
- `approximator`: the type of approximator
- `minimizers_diff_norm_mean`: the mean value of 2-norm of the difference between true and estimated minimizers
- `optvals_diff_abs_mean`: the mean value of absolute of the difference between true and estimated optimal values
- `no_of_minimizer_success_cases`: failure means no minimizer has been found (`NaN`)
- `no_of_optval_success_cases`: failure means invalid optimal value has been found (`-Inf` or `Inf)
- `number_of_parameters`: the number of network parameters


#### Results
- Run as
```julia
include("main/basic.jl")
main(1, 1); main(61, 20); main(376, 17)  # second run
```

- (n, m) = (1, 1)
```julia
 Row │ n      m      epochs  approximator  optimize_time_mean  minimizers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimizer_success  no_of_optval_success  number_of_parameters
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
 Row │ n      m      epochs  approximator  optimize_time_mean  minimizers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimizer_success  no_of_optval_success  number_of_parameters
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
 Row │ n      m      epochs  approximator  optimize_time_mean  minimizers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimizer_success  no_of_optval_success  number_of_parameters
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
- [3] [J. Kim and Y. Kim, “Parameterized Convex Universal Approximators for Decision-Making Problems,” IEEE Trans. Neural Netw. Learning Syst., accepted for publication, 2022, doi: 10.1109/TNNLS.2022.3190198.](https://ieeexplore.ieee.org/document/9833537)
