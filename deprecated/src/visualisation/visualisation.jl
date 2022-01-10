"""
    plot_figure!(fig, approx, xlim, ulim)

Suppose that approx is bivariate function, i.e.,
(x, u) -> approx([x], [u]) where x ∈ ℝ, u ∈ ℝ.
"""
function plot_approx!(fig::AbstractPlot, approx, xlim, ulim;
        Δx=0.1, Δu=0.1, kwargs...)
    @assert xlim |> length == 2
    @assert ulim |> length == 2
    # if the output is vector, extract only the element, i.e., make the output scalar
    func(x, u) = approx([x], [u])[1]
    plot!(fig,
          (xlim[1]):Δx:(xlim[2]), (ulim[1]):Δx:(ulim[2]), func;
          st=:surface,
          kwargs...,
         )
end
