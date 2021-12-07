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
- [x] 최적화 실패 처리 개선
    - 현재는 실패한 경우가 하나만 있어도 아예 데이터에서 제외함
    - 실패한 경우의 케이스 수를 받고, 성공한 녀석들로 평균값을 구하자
- [x] 지금까지의 결과로 논문 시뮬파트 작성
- [ ] Finite-horizon Q-learning 코드 작성
- [ ] (future works) network converter from Julia to Python and vice versa;
for differentiable convex programming
- [ ] (future works) add infinite-horizon Q-learning examples via differentiable convex programming


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
```julia
 Row │ n      m      epochs  approximator  optimise_time_mean  minimisers_diff_norm_mean  optvals_diff_abs_mean  no_of_minimiser_success  no_of_optval_success  number_of_parameters
     │ Int64  Int64  Int64   String        Float64             Float64                    Float64                Int64                    Int64                 Int64
─────┼───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
   1 │     1      1     100  FNN                   0.00136872                 0.0339567              0.00835098                      500                   500                  4417
   2 │     1      1     100  MA                    0.00174339                 0.0517859              0.131502                        500                   500                    90
   3 │     1      1     100  LSE                   0.071346                   0.00626685             0.126809                        500                   500                    90
   4 │     1      1     100  PICNN                 0.0287229                  0.050802               0.100065                        500                   499                 25608
   5 │     1      1     100  PMA                   0.00252154                 0.324986               0.0189862                       500                   500                  8188
   6 │     1      1     100  PLSE                  0.0137478                  0.00257412             0.00157031                      500                   500                  8188
   7 │    13      4     100  FNN                   0.0239815                  0.998098               0.179316                        500                   500                  5377
   8 │    13      4     100  MA                    0.00272083                 0.219982               0.063618                        500                   500                   540
   9 │    13      4     100  LSE                   0.0565649                  0.0561577              0.0360103                       500                   500                   540
  10 │    13      4     100  PICNN                 0.0253631                  0.366856               0.0343946                       500                   500                 27987
  11 │    13      4     100  PMA                   0.00227412                 0.498253               0.015404                        500                   500                 14806
  12 │    13      4     100  PLSE                  0.0159233                  0.0568649              0.00995584                      500                   500                 14806
  13 │   376     17     100  FNN                   0.0786036                  2.94953                0.165373                        500                   500                 29441
  14 │   376     17     100  MA                    0.0173594                  4.10848                0.125641                        500                   500                 11820
  15 │   376     17     100  LSE                   0.316279                   4.01288                0.115364                        500                   500                 11820
  16 │   376     17     100  PICNN                 0.432312                   3.79597                0.132973                        500                   500                 84534
  17 │   376     17     100  PMA                   0.0361832                  3.51644                0.0878767                       500                   500                 63388
  18 │   376     17     100  PLSE                  0.0169323                  0.568872               0.0828379                       500                   500                 63388
```

## References
- [1] [G. C. Calafiore, S. Gaubert, and C. Possieri, “Log-Sum-Exp Neural Networks and Posynomial Models for Convex and Log-Log-Convex Data,” IEEE Transactions on Neural Networks and Learning Systems, vol. 31, no. 3, pp. 827–838, Mar. 2020, doi: 10.1109/TNNLS.2019.2910417.](https://ieeexplore.ieee.org/abstract/document/8715799?casa_token=ptHxee1NJ30AAAAA:etAIY0UkR0yg6YK7mgtEzCzHavM0d6Cos1VNzpn0cw5hbiEnFnAxNDm1rflWjDAOa-iO6xU5Lg)
