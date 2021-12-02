# ParametrisedConvexApproximators

[ParametrisedConvexApproximators.jl](https://github.com/JinraeKim/ParametrisedConvexApproximators.jl) is a Julia package providing predefined pararmetrised convex approximators and related functionalities.

## Installation
To install ParametrisedConvexApproximator,
please open Julia's interactive session (known as REPL) and press `]` key
in the REPL to use the package mode, then type the following command

```julia
pkg> add ParametrisedConvexApproximator
```

## Quick Start
- Activate multi-threading if available, e.g., `julia -t 7` enabling `7` threads.
It will reduce computation time for minimising networks w.r.t. multiple points.
- ParametrisedConvexApproximators.jl focuses on providing predefined approximators including parametrised convex approximators.
Note that when approximators receive two arguments, the first and second arguments correspond to
condition and decision vectors, usually denoted by `x` and `u`.
- Construction and optimisation of parametrised log-sum-exp (PLSE) approximator:
```julia
using ParametrisedConvexApproximators
using Flux

# construction
n, m = 2, 3
i_max = 20
T = 1e-1
h_array = [128, 128]
act = Flux.leakyrelu
plse = PLSE(n, m, i_max, T, h_array, act)
# optimisation
x = rand(n)
u_min, u_max = -1*ones(m), 1*ones(m)
res = optimise(plse, x; u_min=u_min, u_max=u_max)  # minimsation
@show res  # NamedTuple
```

```julia
res = (minimiser = [-0.2523565154854893, -0.9999967116995178, 0.09150518836473269], optval = [0.2943142110436148])
```
- The approximators can be trained via [Flux.jl](https://github.com/FluxML/Flux.jl), an ML library of Julia.
    - [ ] To-do; add a Flux.jl example

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
    - `PMA::ParametrisedConvexApproximator`: parametrised MA network
    - `PLSE::ParametrisedConvexApproximator`: parametrised LSE network
    - `PICNN::ParametrisedConvexApproximator`: partially input-convex neural network

### Utilities
- `(nn::approximator)(x, u)` gives an inference (approximate function value).
- `res = optimise(approximator, x; u_min=nothing, u_max=nothing)` provides
minimiser and optimal value (optval) for given `x` as `res.minimiser` and `res.optval`
considering box constraints of `u >= u_min` and `u <= u_max` (element-wise).
    - The condition variable `x` can be a vector, i.e., `size(x) = (n,)`,
    or a matrix for parallel solve (via multi-threading), i.e., `size(x) = (n, d)`.


## Notes
### To-do list
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
- [x] FNN 구현
- [x] 테스트 목록 만들기
    - 기본 기능 체크 (네트워크 생성, inference dimension, etc.)
    - Inference speed -> x
    - 최적화 속도
- [x] PLSE 구현
    - NN 을 하나 넣는 것과 두 개 넣는 것, 두 버전을 고려해야할수도 있다.
    - 일단 하나 넣는거로 구현하고, 이후 수정이 필요하다 생각되면 "변경이 가능하게" 하자.
- [x] LSE 구현
- [x] PMA 구현
- [x] MA 구현
- [x] 함수 근사 학습 코드 작성
    - 일단은 최대한 normalisation 등 없이 pure 하게
    - [-1, 1]^n 에서 샘플링
- [x] Convex opt. solver 결정
    - SCS.jl 로 결정; open source
- [x] 최적화 정확도 체크 (minimiser, optval)
- [x] 최적화 솔버 재결정 (Julia 1.7 과 SCS.jl 가 호환 안 됨; Mosek 은 M1 native 로 안 돌아가는듯; 현재는 COSMO.jl 임)
    - 일단 COSMO.jl 로 결정함
- [x] PICNN 구현
    - [x] gradient projection 주의
- Visualisation
    - [x] minimiser diff norm and optval diff abs (histogram)
    - [x] surface (for n = 1, m = 1)
- [ ] 최적화 실패 처리 개선
    - 현재는 실패한 경우가 하나만 있어도 아예 데이터에서 제외함
    - 실패한 경우의 케이스 수를 받고, 성공한 녀석들로 평균값을 구하자
- [ ] 지금까지의 결과로 논문 시뮬파트 작성
- [ ] Finite-horizon Q-learning 코드 작성
- [ ] (future works) network converter from Julia to Python and vice versa;
for differentiable convex programming
- [ ] (future works) add infinite-horizon Q-learning examples via differentiable convex programming


### Benchmark
The following result is from `test/basic.jl`.
- `n`: dimension of condition variable `x`
- `m`: dimension of decision variable `u`
- `epochs`: training epochs
- `optimisation_failure`: if there is at least one failure case in optimisation, `true`
- `minimisers_diff_norm_mean`: the mean value of 2-norm of the difference between true and estimated minimisers
- `optvals_diff_abs_mean`: the mean value of absolute of the difference between true and estimated optimal values
- `optimise_time_mean`: average time for optimisation
- `no_of_optimise_points`: number of optimisation points to obtain benchmark results, using [BenchmarkTools.jl](https://github.com/JuliaCI/BenchmarkTools.jl) (can be truncated for too slow optimisation)
```julia
 Row │ optimise_time_mean  no_of_optimise_points  minimisers_diff_norm_mean  optvals_diff_abs_mean  n      m      epochs  approximator  optimisation_failure
     │ Float64?            Union{Missing, Int64}  Float64?                   Float64?               Int64  Int64  Int64   String        Bool
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │         0.00215083                    200                 0.0307585              0.00891506      1      1     100  FNN                          false
   2 │         0.00171477                    200                 0.337122               0.125028        1      1     100  MA                           false
   3 │         0.0336831                     200                 0.0607105              0.126509        1      1     100  LSE                          false
   4 │         0.134263                      200                 0.0512199              0.0904881       1      1     100  PICNN                        false
   5 │         0.0015151                     200                 0.321775               0.0125168       1      1     100  PMA                          false
   6 │         0.00870789                    200                 0.00837253             0.00732863      1      1     100  PLSE                         false
   7 │         0.102924                      200                 2.1967                 0.593645       10     10     100  FNN                          false
   8 │         0.0264953                     200                 2.65532                0.318628       10     10     100  MA                           false
   9 │         0.042978                      200                 2.35952                0.263024       10     10     100  LSE                          false
  10 │         0.375921                       80                 2.77599                0.178636       10     10     100  PICNN                        false
  11 │         0.00569223                    200                 1.7945                 0.173026       10     10     100  PMA                          false
  12 │         0.0130083                     200                 1.58341                0.264676       10     10     100  PLSE                         false
  13 │         1.18958                        26                 7.16768                1.66083       100    100     100  FNN                          false
  14 │         0.145114                      200                 9.60918                0.197156      100    100     100  MA                           false
  15 │         0.0653175                     200                 9.64556                0.167201      100    100     100  LSE                          false
  16 │         3.3229                         10                 9.94982                0.917935      100    100     100  PICNN                        false
  17 │         0.116434                      200                 9.55993                0.265828      100    100     100  PMA                          false
  18 │         0.0670418                     200                 9.62691                0.431962      100    100     100  PLSE                         false
```

## References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
