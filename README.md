# ParametrisedConvexApproximators

[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl) is a package providing (pararmetrised) convex approximators.

## Supported approximators
This package provides the following approximators. 
- **convex approximators (CAs)**
    - MA (max-affine) [1]
    - LSE (log-sum-exp) [1]

- **parametrised convex approximators (PCAs)** (a generalised concept of convex approximators)
    - PMA (parametrised max-affine)
    - PLSE (parametrised log-sum-exp)

# Usage patterns
## Data
### `AbstractDataStructure`
- Construct a `data::AbstractDataStructure` for convenient data manipulation. For example, the following example constructs a `data::xufData`.
```julia
n, m, d = 1, 1, 1000
xlim = (-5, 5)
ulim = (-5, 5)
xs = 1:d |> Map(i -> xlim[1] .+ (xlim[2]-xlim[1]) .* rand(n)) |> collect
us = 1:d |> Map(i -> ulim[1] .+ (ulim[2]-ulim[1]) .* rand(m)) |> collect
fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
xuf_data = xufData(xs, us, fs)
```
### `AbstractNormaliser`
- Construct a `normaliser::AbstractNormaliser` for data normalisation.
    - I would recommend you to use `MinMaxNormaliser`, which normalises data into a hypercube of `[0, 1]^l`.


## Approximator
### `AbstractApproximator`
- Construct an `approximator::AbstractApproximator`.
- It is NOT recommended to use `approximator::AbstractApproximator` directly. Instead, use `NormalisedApproximator` for better performance of training.

## `NormalisedApproximator`
- Construct a `normalised_approximator::NormalisedApproximator` instead of `approximator::AbstractApproximator`.
It is recommended you to use normalised approximator due to the training performance; this will automatically normalise and unnormalise given mini-batch data. Normalisation usually gives a better result.
For example, the following example constructs a `normalised_approximator::NormalisedApproximator` based on `approximator::PLSE` and `normaliser::MinMaxNormaliser`.
```julia
n, m, d = length(xuf_data.x[1]), length(xuf_data.u[1]), xuf_data.d
i_max = 20
# h_array = [16, 16]
h_array = [64, 64, 64]
T = 1e-1
act = Flux.leakyrelu
approximator = PLSE(n, m, i_max, T, h_array, act)
normalised_approximator = NormalisedApproximator(plse, MinMaxNormaliser(xuf_data)),  # Note: MinMaxNormaliser is better than StandardNormalDistributionNormaliser
```

## Training
### Losses
- [ ] To-do

### Training methods
- [ ] To-do



# To-do list
- [x] Add `solve!` test.
- [x] Add normalisation
- [ ] Add Q-learning algorithm
- [ ] Apply Q-learinng to optimal control with non-convex value function using PCAs.
- [ ] quantitative study of supervised learning; accuracy, computation speed, etc.
- [ ] quantitative study of Q-learning; performance index, etc.
- [ ] Write academic papers!

## Not urgent
- [ ] Flux-based AD (auto differentiation)-compatible construction for PMA. See 
`PMA(n::Int, m::Int, u_is::Vector, u_star_is::Vector, f::Function)`.


# References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
