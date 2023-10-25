using Plots
using LaTeXStrings
using Random


Random.seed!(2023)
N = 10
u_rands = [(2*rand(1)[1] - 1) for i in 1:N]


function target(x, u)
  return -x^2 + u^2
end

function target_gradient(x, u)
  2*u
end

function target_abs(x, u)
  return -x^2 + abs(u)
end


function visualize_MA(u_rands)
  anim = Animation()
  us = -1:0.05:1

  for (idx, u_rand) in enumerate(u_rands)
    fig = plot(;
               xlabel=L"$u$",
               ylabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               legend=:topleft,
               dpi=300,
               size=(600, 600),
               label=nothing,
              )
    plot!(us, u->target(0, u), alpha=0.5, lc=:red, label=nothing)

    function ma_func(x, u)
      maximum([target(0, u_rands[i]) + target_gradient(0, u_rands[i]) * (u - u_rands[i])] for i in 1:idx-1)[1]
    end
    if idx != 1
      plot!(
            us, u -> ma_func(0, u); lc=:blue, alpha=0.2,
            label=nothing,
           )
      frame(anim)
    end

    plot!(
          u_rand * ones(length(us)), LinRange(-1, target(0, u_rand), length(us));
          lc=:black,
          label=nothing,
         )
    frame(anim)
    function MA_tmp(x, u)
      target(0, u_rands[idx]) + target_gradient(0, u_rands[idx]) * (u - u_rands[idx])
    end
    plot!(
          us, u -> MA_tmp(0, u); alpha=0.5,
          lc=:red,
          label=nothing,
         )
    frame(anim)
  end
  gif(anim, "anim_ma.gif"; fps=2,
      # loop=100,
     )
end


function visualize_MA_subgrad()
  Random.seed!(2023)
  anim = Animation()
  us = -1:0.05:1
  u_grads = [2*(rand(1)[1]-0.5) for i in 1:length(us)]
  anim = Animation()


  for (idx, u_grad) in enumerate(u_grads)
    fig = plot(;
               xlabel=L"$u$",
               ylabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               legend=:topleft,
               dpi=300,
               size=(600, 600),
               label=nothing,
              )
    plot!(us, u->target_abs(0, u), alpha=0.5, lc=:red, label=nothing)

    plot!(
          0 * ones(length(us)), LinRange(-1, target(0, 0), length(us));
          lc=:black,
          label=nothing,
         )
    function MA_tmp(x, u)
      target(0, 0) + u_grad * (u - 0)
    end
    plot!(
          us, u -> MA_tmp(0, u); alpha=0.5,
          lc=:purple,
          label=nothing,
         )
    frame(anim)
  end
  gif(anim, "anim_ma_subgrad.gif"; fps=2,
     )
end


function visualize_PMA(u_rands)
  anim = Animation()
  xs = -1:0.1:1
  us = -1:0.1:1

  for (idx, u_rand) in enumerate(u_rands)
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
      maximum([target(x, u_rands[i]) + target_gradient(x, u_rands[i]) * (u - u_rands[i])] for i in 1:idx-1)[1]
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
      target(x, u_rands[idx]) + target_gradient(x, u_rands[idx]) * (u - u_rands[idx])
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


function visualize_PMA_subgrad(; seed=2023)
  Random.seed!(2023)
  anim = Animation()
  xs = -1:0.05:1
  us = -1:0.05:1
  coeff_rands = [rand(1)[1] for i in 1:length(us)]
  u_grads = [x -> coeff_rands[i]*x for i in 1:length(us)]

  for (idx, u_grad) in enumerate(u_grads)
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
               label=nothing,
               st=:surface,
               colorbar=false,
              )
    plot!(xs, us, target_abs, alpha=0.5, lc=:red, label=nothing,
          st=:surface,
          colorbar=false,
         )


    plot!(
          xs, repeat([0], length(xs)), target_abs.(xs, repeat([0], length(xs)));
          lc=:black,
          label=nothing,
         )
    function PMA_tmp(x, u)
      target_abs(x, 0) + u_grad(x) * (u - 0)
    end
    plot!(
          xs, us, PMA_tmp; st=:surface, color=:blue, alpha=0.5, colorbar=false,
         )
    frame(anim)
  end
  gif(anim, "anim_pma_subgrad.gif"; fps=2,
     )
end



function visualize_PLSE(u_rands)
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
                      ) for u_rand in u_rands]
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


function visualize()
  visualize_PMA(u_rands)
  visualize_PLSE(u_rands)
end
