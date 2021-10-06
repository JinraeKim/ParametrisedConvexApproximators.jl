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


# To-do list
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
