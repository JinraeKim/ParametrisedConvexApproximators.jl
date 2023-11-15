using Plots
using LaTeXStrings
using Random


Random.seed!(2023)
N = 10
N_MA_subgrads = 11
N_PMA_subgrads = 41
u_rands = [(2*rand(1)[1] - 1) for i in 1:N]


function target(x, u)
  return -x^2 + u^2
end

function target_gradient(x, u)
  2*u
end

function target_abs(x, u)
  return abs(u - x)
end


function target_abs_gradient(x, u; eps=1e-3)
  if u - x > eps
    grad = 1
  elseif u - x < -eps
    grad = -1
  else
    grad = 0
  end
end


function visualize_MA(u_rands)
  anim = Animation()
  us = -1:0.01:1

  for (idx, u_rand) in enumerate(u_rands)
    fig = plot(;
               xlabel=L"$u$",
               ylabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               legend=:topleft,
               dpi=300,
               size=(450, 450),
               label=nothing,
              )
    plot!(us, u->target(0, u), alpha=0.5, lc=:red, label=nothing,
          lw=2,
         )

    function ma_func(x, u)
      maximum([target(0, u_rands[i]) + target_gradient(0, u_rands[i]) * (u - u_rands[i])] for i in 1:idx-1)[1]
    end
    if idx != 1
      plot!(
            us, u -> ma_func(0, u); lc=:blue, alpha=0.2,
            label=nothing,
            lw=2,
           )
      frame(anim)
    end

    plot!(
          u_rand * ones(length(us)), LinRange(-1, target(0, u_rand), length(us));
          lc=:black,
          ls=:dash,
          label=nothing,
          lw=2,
         )
    frame(anim)
    function MA_tmp(x, u)
      target(0, u_rands[idx]) + target_gradient(0, u_rands[idx]) * (u - u_rands[idx])
    end
    plot!(
          us, u -> MA_tmp(0, u); alpha=0.5,
          lc=:purple,
          label=nothing,
          lw=2,
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
  us = -1:0.01:1
  u_grads = [2*(rand(1)[1]-0.5) for i in 1:N_MA_subgrads]
  anim = Animation()


  for (idx, u_grad) in enumerate(u_grads)
    println("$(idx)/$(length(u_grads))")
    fig = plot(;
               xlabel=L"$u$",
               ylabel=L"$f$",
               xlim=(-1, 1),
               ylim=(-1, 1),
               legend=:topleft,
               dpi=300,
               size=(450, 450),
               label=nothing,
              )
    plot!(us, u->target_abs(0, u), alpha=0.5, lc=:red, label=nothing,
          lw=2,
         )

    plot!(
          0 * ones(length(us)), LinRange(-1, target(0, 0), length(us));
          lc=:black,
          ls=:dash,
          label=nothing,
          lw=2,
         )
    function MA_tmp(x, u)
      target(0, 0) + u_grad * (u - 0)
    end
    plot!(
          us, u -> MA_tmp(0, u); alpha=0.5,
          lc=:purple,
          label=nothing,
          lw=2,
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
               zlim=(-2, 2),
               legend=:topleft,
               dpi=300,
               size=(450, 450),
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
          ls=:dash,
          label=nothing,
         )
    frame(anim)
    function PMA_tmp(x, u)
      target(x, u_rands[idx]) + target_gradient(x, u_rands[idx]) * (u - u_rands[idx])
    end
    plot!(
          xs, us, PMA_tmp; st=:surface, color=:purple, alpha=0.5, colorbar=false,
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
  xs = -1:0.01:1
  us = -1:0.01:1
  phases = range(0, stop=2*pi, length=N_PMA_subgrads)
  # xs = [-1 + 1*(1+cos(phase)) for phase in phases]
  # u_grads = [target_abs_gradient(x, 0) for x in xs]

  for (i, phase) in enumerate(phases)
    _x = -1 + 1*(1+cos(phase))
    u_grad = target_abs_gradient(_x, 0)
    println("$(i)/$(length(phases))")
    l = @layout [
                 a{0.6w} b
                ]
    fig = plot(;
               layout=l,
               size=(450*2, 450),
               dpi=300,
               legend=:topleft,
              )
    plot!(
          fig[1];
          xlabel=L"$x$",
          ylabel=L"$u$",
          zlabel=L"$f$",
          xlim=(-1, 1),
          ylim=(-1, 1),
          zlim=(-2, 2),
          label=nothing,
          st=:surface,
          colorbar=false,
         )
    plot!(
          fig[1],
          xs, us, target_abs, alpha=0.5, lc=:red,
          label=nothing,
          st=:surface,
          colorbar=false,
         )
    z = map(target_abs, _x * ones(length(us)), us)
    plot!(
          fig[1],
          _x * ones(length(us)), us, z, line=(:red, 5, 0.2),
          label=nothing,
         )

    plot!(
          fig[1],
          xs, repeat([0], length(xs)), target_abs.(xs, repeat([0], length(xs)));
          lc=:black,
          ls=:dash,
          label=nothing,
         )
    plot!(
          fig[2],
          us, u -> target_abs(_x, u);
          color=:red, alpha=0.5,
          xlim=(-1, 1),
          ylim=(-2, 2),
          xlabel=L"$u$",
          ylabel=L"$f$",
          label=nothing,
          lw=2,
          # st=:surface,
          # colorbar=false,
         )
    plot!(
          fig[2],
          0 * ones(length(us)), LinRange(-2, target_abs(_x, 0), length(us));
          lc=:black,
          ls=:dash,
          label=nothing,
          lw=2,
         )
    PMA_tmp = [target_abs(_x, 0) + u_grad * (u-0) for u in us]
    plot!(
          fig[2],
          us, PMA_tmp;
          color=:purple, alpha=0.5,
          label=nothing,
          lw=2,
          subplot=2,
          # xlabel=L"$u$",
          # ylabel=L"$f$",
          # st=:surface,
          # colorbar=false,
         )
    frame(anim)
  end
  gif(anim, "anim_pma_subgrad.gif"; fps=10,
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
               zlim=(-2, 2),
               legend=:topleft,
               dpi=300,
               size=(450, 450),
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


function visualize_continuous_approximate_selection()
  fig = plot(;
             aspect_ratio=1,
            )
  xs1 = -1:0.01:0
  xs2 = 0:0.01:+1
  plot!(xs1, -1*ones(length(xs1));
        label="multivalued function",
        line=(:red, 2),
       )
  plot!(xs2, +1*ones(length(xs2));
        label=nothing,
        line=(:red, 2),
       )
  plot!(zeros(length(xs1)), LinRange(-1, 1, length(xs1));
        label=nothing,
        line=(:red, 2),
       )
  xs = [-1.1, +0.1, +0.1, +1.1, +1.1, -0.1, -0.1, -1.1]
  ys = [-1.1, -1.1, +0.9, +0.9, +1.1, +1.1, -0.9, -0.9]
  plot!(
        Shape(xs, ys),
        st=:shape,
        label=nothing,
        alpha=0.2,
        color=:black,
       )
  sigmoid = function (x; a=35)
    2*((1 / (1+exp(-a*x))) - 0.5)
  end
  plot!(
        [xs1..., xs2...], sigmoid,
        label=L"$\epsilon$" * "-selection",
        line=(:blue, :dash, 2)
       )
  savefig("continuous_approximate_selection.pdf")
  savefig("continuous_approximate_selection.png")
end


function visualize()
  visualize_PMA(u_rands)
  visualize_PLSE(u_rands)
end

