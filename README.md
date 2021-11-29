# ParametrisedConvexApproximators

[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl) is a package providing (pararmetrised) convex approximators.

## Quick Start
- Activate multi-threading if available, e.g., `julia -t 7`.
It will reduce computation time for minimising networks w.r.t. multiple points.

## NOTICE: the source code will be rewritten.
### Why...?
The previous code is outdated, and too complicated to be modified.
For example, I have no idea why convex solvers take much longer time than non-convex solvers,
which is opposite to previously presentation in lab seminar, FDCL, SNU by Jinrae Kim.

## To-do list
- [x] 최적화 방식 결정 (decide how to write optimisation code)
    - `test/solver.jl` 참고 (see the file)
    - convex 솔버는 Convex.jl + Mosek (convex solver is...)
    - non-convex 솔버는 IPNewton in Optim.jl (non-convex solver is...)
    - opt variable, opt problem 등은 기존과 같이 내부적으로 매번 새로 생성 (opt. variable and opt. problem will be re-created every time the optimisation problem is solved.)
    <details>
    <summary>차원이 증가함에 따라 convex 솔버가 scalable 함을 확인 (convex solver is scalable)</summary>

    ```julia
    (n, m) = (N, N) = (1, 1)
    convex solver
      690.750 μs (4145 allocations: 253.30 KiB)
    non-convex solver (ipnewton)
      200.917 μs (3817 allocations: 245.88 KiB)
    (n, m) = (N, N) = (10, 10)
    convex solver
      1.202 ms (8788 allocations: 588.77 KiB)
    non-convex solver (ipnewton)
      956.000 μs (14722 allocations: 2.12 MiB)
    (n, m) = (N, N) = (100, 100)
    convex solver
      5.885 ms (54237 allocations: 3.92 MiB)
    non-convex solver (ipnewton)
      198.712 ms (983575 allocations: 856.71 MiB)
    ```

    </details>
    - Note: 솔버가 서로 달라서, 절대적인 시간도 중요하지만 차원의 증가에 따른 소요 시간의 변화가 더 중요
    ('cause the solvers are different, we need to focus on "how much the elapsed time increases as the dimension gets higher")
- [x] Deprecated code 는 한데 모으기
    - See `./deprecated`.
- [ ] FNN 구현
- [ ] 테스트 목록 만들기
    - 기본 기능 체크 (네트워크 생성, inference dimension, etc.)
    - Inference speed
    - 최적화 속도
- [ ] LSE 구현
- [ ] PLSE 구현
    - NN 을 하나 넣는 것과 두 개 넣는 것, 두 버전을 고려해야할수도 있다.
    - 일단 하나 넣는거로 구현하고, 이후 수정이 필요하다 생각되면 "변경이 가능하게" 하자.
- [ ] 함수 근사 학습 코드 작성
    - 일단은 최대한 normalisation 등 없이 pure 하게
    - [-1, 1]^n 에서 샘플링
- [ ] Finite-horizon Q-learning 코드 작성




# !The followings are deprecated!
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
- `AbstractApproximator` includes
    - (`ConvexApproximator`) `MA`, `LSE`
    - (`ParametrisedConvexApproximator`) `PMA`, `PLSE`

### `NormalisedApproximator`
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
