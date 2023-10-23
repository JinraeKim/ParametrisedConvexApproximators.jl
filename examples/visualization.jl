using Plots
using LaTeXStrings
using Random


function target(x, u)
  return -x^2 + u^2
end

function target_gradient(x, u)
  2*u
end


function visualize_PMA(us_rand)
  anim = Animation()
  xs = -1:0.1:1
  us = -1:0.1:1

  for (idx, u_rand) in enumerate(us_rand)
    fig = plot(;
               xlabel=L"$x$",
               ylabel=L"$u$",
               zlabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               zlim=(-1, 1),
               legend=:topleft,
               dpi=300,
               size=(600, 600),
              )
    plot!(xs, us, target, st=:surface, alpha=0.5, colorbar=false)

    function pma_func(x, u)
      maximum([target(x, us_rand[i]) + target_gradient(x, us_rand[i]) * (u - us_rand[i])] for i in 1:idx-1)[1]
    end
    if idx != 1
      plot!(
            xs, us, pma_func; st=:surface, color=:blue, alpha=0.2, colorbar=false,
           )
      frame(anim)
    end

    plot!(
          xs, repeat([u_rand], length(xs)), target.(xs, repeat([u_rand], length(xs)));
          lc=:black,
          label=nothing,
         )
    frame(anim)
    function PMA_tmp(x, u)
      target(x, us_rand[idx]) + target_gradient(x, us_rand[idx]) * (u - us_rand[idx])
    end
    plot!(
          xs, us, PMA_tmp; st=:surface, alpha=0.5, colorbar=false,
          # label=L"$f(x, u_{i}) + \langle \hat{u}^{*}_{\epsilon, i}, u - u_{i} \rangle$",
         )
    frame(anim)
  end
  gif(anim, "anim_pma.gif"; fps=2,
      # loop=100,
     )
end


function visualize_PLSE(us_rand)
  anim = Animation()
  xs = -1:0.1:1
  us = -1:0.1:1
  Ts = [10.0^(-i) for i in -1:0.5:5]

  for T in Ts
    fig = plot(;
               xlabel=L"$x$",
               ylabel=L"$u$",
               zlabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               zlim=(-1, 1),
               legend=:topleft,
               size=(600, 600),
              )
    plot!(xs, us, target, st=:surface, alpha=0.5, colorbar=false)

    function plse_func(x, u)
      T * log(
              sum(
                  [exp(
                       (target(x, u_rand) + target_gradient(x, u_rand) * (u - u_rand)) / T
                      ) for u_rand in us_rand]
                 )
             )
    end
    plot!(
          xs, us, plse_func; st=:surface, color=:blue, alpha=0.2, colorbar=false,
         )
    annotate!(-1, 1, 1, (L"$T = 10^{%$(log10(T))}$", 40))
    frame(anim)
  end
  gif(anim, "anim_plse.gif"; fps=2,
      # loop=100,
     )
end


function visualize(; seed=2023, N=10)
  Random.seed!(seed)
  us_rand = [(2*rand(1)[1] - 1) for i in 1:N]

  # visualize_PMA(us_rand)
  visualize_PLSE(us_rand)
end
