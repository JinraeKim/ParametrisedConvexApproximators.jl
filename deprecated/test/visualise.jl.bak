using ParametrisedConvexApproximators
const PCApprox = ParametrisedConvexApproximators

using Transducers
using Flux
using Random
using Plots
using LaTeXStrings


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

"""
See this:
https://stackoverflow.com/questions/66417677/3d-surface-plot-in-julia
"""
function plot_figures!(fig, func; kwargs...)
    plot!(fig,
          -1:0.1:1, -1:0.1:1, func;
          st=:surface,
          kwargs...
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
    fig_f = plot(;
                 xlim=(-1, 1),
                 ylim=(-1, 1),
                 zlim=(-1, 1),
                 aspect_ratio=:equal,
                )
    fig_diff = plot(;
               xlim=(-1, 1),
               ylim=(-1, 1),
               zlim=1e-2.*(-1, 1),
               aspect_ratio=:equal,
              )
    func_diff(x, u) = pma([x], [u]) - f(x, u)
    plot_figures!(fig_diff, func_diff; title=L"\hat{f} - f")
    plot_figures!(fig_f, f; title=L"f")
    mkpath("figures")
    fig = plot(fig_f, fig_diff; layout=(2, 1), size=(800, 800))
    savefig(fig, "figures/vis.pdf")
    nothing
end
