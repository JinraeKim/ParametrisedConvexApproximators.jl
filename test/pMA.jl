using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Transducers
using Flux
using Random
using Plots


function f(x, u)
    0.5 * (-x'*x + u'*u)
end

function f_partial_u(x, u)
    u
end

function initialise(n, m, d)
    xs = 1:d |> Map(i -> -1 .+ 2*rand(n)) |> collect
    us = 1:d |> Map(i -> -1 .+ 2*rand(m)) |> collect
    fs = zip(xs, us) |> MapSplat((x, u) -> f(x, u)) |> collect
    u_is = -1:0.1:1 |> Map(_u_i -> [_u_i]) |> collect  # to make it a matrix
    u_star_is = u_is |> Map(u_i -> (x -> f_partial_u(x, u_i))) |> collect
    # constructed
    pma = PCApprox.pMA(n, m, u_is, u_star_is, f)
    @show u_is |> length == pma.i_max
    @show pma.i_max
    # infer
    @time @show pma.NN(rand(n)) |> size
    @show pma(rand(n), rand(m)) |> size
    @show pma(rand(n, d), rand(m, d)) |> size
    # normal
    i_max = pma.i_max
    _pma = PCApprox.pMA(n, m, i_max, Dense(n, i_max*(m+1)))
    @show _pma.NN(rand(n)) |> size
    @show _pma(rand(n), rand(m)) |> size
    @show _pma(rand(n, d), rand(m, d)) |> size
    #
    xu_data = PCApprox.xuData(xs, us)
    pma, xu_data
end

function plot_figures(pma, xu_data)
    xs = xu_data.x
    us = xu_data.u
    fig = plot(;
               xlim=(-1, 1),
               ylim=(-1, 1),
               zlim=1e-2.*(-1, 1),
              )
    @show hcat(xs...)' |> size
    @show pma(xs[1], us[1])
    @show f(xs[1], us[1])
    plot!(fig,
          hcat(xs...)'[:], hcat(us...)'[:], (x, u) -> f(x, u) - pma([x], [u]);  # to make it compatible with st=:surface
          st=:surface,
          label="approx",
         )
    display(fig)
    nothing
end

function main()
    Random.seed!(2021)
    n = 1
    m = 1
    d = 100
    pma, xu_data = initialise(n, m, d)
    plot_figures(pma, xu_data)
end
