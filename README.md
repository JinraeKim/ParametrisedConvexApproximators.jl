# ParametrisedConvexApproximators
[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl)
is a Julia package providing predefined parametrised convex approximators and related functionalities.
An official package of [3].


## Installation
To install ParametrisedConvexApproximator,
please open Julia's interactive session (a.k.a REPL) and press `]` key
in the REPL to use the package mode, then type the following command

```julia
pkg> add ParametrisedConvexApproximator
```

### Notes
- For PLSE(plus), the differentiation of the minimiser is now available via implicit differentiation.
  - Just use `minimise` as before :>
- The benchmark result was reported in [ParametrisedConvexApproximator.jl v0.1.1](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl/tree/v0.1.1) [3].


## Quick Start
ParametrisedConvexApproximators.jl focuses on providing predefined approximators
including parametrised convex approximators.
Note that when approximators receive two arguments, the first and second arguments correspond to
condition and decision vectors, usually denoted by `x` and `u` (or, `c` and `d`), respectively.

### Network construction
```julia
using ParametrisedConvexApproximators
using Flux
using Random  # to reproduce the following result

# construction
seed = 2023
Random.seed!(seed)
n, m = 3, 2
i_max = 20
T = 1.0
h_array = [64, 64]
act = Flux.leakyrelu
network = PLSE(n, m, i_max, T, h_array, act)  # parametrised log-sum-exp (PLSE) network
x, u = rand(n), rand(m)
f̂ = network(x, u)
@show f̂
```

```julia
f̂ = [3.029994811790289]
```

### Prepare dataset
```julia
min_condition = -ones(n)
max_condition = +ones(n)
min_decision = -ones(m)
max_decision = +ones(m)
target_function_name = :quadratic
target_function = example_target_function(target_function_name)  # f(x, u) = x'*x + u'*u
N = 5_000

dataset = DecisionMakingDataset(
    target_function;
    target_function_name=:quadratic,  # just for metadata
    N=N, n=n, m=m, seed=seed,
    min_condition=min_condition,
    max_condition=max_condition,
    min_decision=min_decision,
    max_decision=max_decision,
)
```

### Network training
```julia
trainer = SupervisedLearningTrainer(dataset, network; optimiser=Adam(1e-4))

@show get_loss(trainer, :train)
@show get_loss(trainer, :validate)
Flux.train!(trainer; epochs=200)
@show get_loss(trainer, :test)
```

```julia
get_loss(trainer, :train) = 2.2437410545162115
get_loss(trainer, :validate) = 2.2849741432452193

...

epoch: 199/200
loss_train = 0.0001664964550015733
loss_validate = 0.0003002414225961646
Best network found!
minimum_loss_validate = 0.0003002414225961646
epoch: 200/200
loss_train = 0.0001647995673689787
loss_validate = 0.00029825480495257375
Best network found!
minimum_loss_validate = 0.00029825480495257375

get_loss(trainer, :test) = 0.00026774267336425705
```

### Conditional decision making via optimization (given `x`, find a minimizer `u` and optimal value)
```julia
# optimization
Random.seed!(seed)
x = rand(n)  # any value
minimiser = minimise(network, x; u_min=min_decision, u_max=max_decision)  # box-constrained minimization; you can define your own optimization problem manually.
@show minimiser
@show network(x, minimiser)
@show dataset[:train].metadata.target_function(x, minimiser)
```

```julia
minimiser = [-0.003060366520019827, 0.007150205329478883]
network(x, minimiser) = [0.9629849722035002]
(dataset[:train]).metadata.target_function(x, minimiser) = 0.9666740244969058
```



## Documentation
### Types
- `AbstractApproximator` is an abstract type of approximator.
- `ParametrisedConvexApproximator <: AbstractApproximator` is an abstract type of parametrised convex approximator.
- `ConvexApproximator <: ParametrisedConvexApproximator` is an abstract type of convex approximator.
- `DifferenceOfConvexApproximator <: AbstractApproximator` is an abstract type of difference of convex approximator.

### Approximators
- All approximators in ParametrisedConvexApproximators.jl receive two arguments, namely, `x` and `u`.
When `x` and `u` are vectors whose lengths are `n` and `m`, respectively,
the output of an approximator is **one-length vector**.
    - Note that `x` and `u` can be matrices, whose sizes are `(n, d)` and `(m, d)`,
    for evaluations of `d` pairs of `x`'s and `u`'s.
    In this case, the output's size is `(1, d)`.

- The list of predefined approximators:
    - `FNN::AbstractApproximator`: feedforward neural network
    - `MA::ConvexApproximator`: max-affine (MA) network [1]
    - `LSE::ConvexApproximator`: log-sum-exp (LSE) network [1]
    - `PICNN::ParametrisedConvexApproximator`: partially input-convex neural network (PICNN) [2]
    - `PMA::ParametrisedConvexApproximator`: parametrised MA (PMA) network [3]
    - `PLSE::ParametrisedConvexApproximator`: parametrised LSE (PLSE) network [3]
        - The default setting is `strict = false`.
        - `PLSEPlus` = `PLSE` with `strict=true`
    - `DLSE::DifferenceOfConvexApproximator`: difference of LSE (DLSE) network [4]

### Interface
- `(nn::approximator)(x, u)` gives an inference (approximate function value).
- `minimiser = minimise(approximator, x; u_min=nothing, u_max=nothing)` provides the minimiser for given condition `x`
considering box constraints of `u >= u_min` and `u <= u_max` (element-wise).
    - The condition variable `x` can be a vector, i.e., `size(x) = (n,)`,
    or a matrix for multiple conditions via multi-threading, i.e., `size(x) = (n, d)`.

### Dataset
- `DecisionMakingDataset`

### Trainer
- `SupervisedLearningTrainer`



## References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
- [2] [B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in Proceedings of the 34th International Conference on Machine Learning, Sydney, Australia, Jul. 2017, pp. 146–155.](http://proceedings.mlr.press/v70/amos17b.html)
- [3] [J. Kim and Y. Kim, “Parameterized Convex Universal Approximators for Decision-Making Problems,” IEEE Trans. Neural Netw. Learning Syst., accepted for publication, 2022, doi: 10.1109/TNNLS.2022.3190198.](https://ieeexplore.ieee.org/document/9833537)
- [4] [G. C. Calafiore, S. Gaubert, and C. Possieri, “A Universal Approximation Result for Difference of Log-Sum-Exp Neural Networks,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 12, pp. 5603–5612, Dec. 2020, doi: 10.1109/TNNLS.2020.2975051.](https://ieeexplore.ieee.org/abstract/document/9032340)
