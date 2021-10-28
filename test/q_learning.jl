using ParametrisedConvexApproximators
const PCA = ParametrisedConvexApproximators
using FlightSims
using FSimZoo
const FS = FlightSims
using UnPack
using DataFrames
# using FSimZoo
using Transducers
using Plots
using Convex
using Mosek, MosekTools
using Flux
using Random
using ComponentArrays
using LinearAlgebra
using LaTeXStrings


function sample(lim)
    @assert length(lim) == 2
    l = length(lim[1])
    @assert length(lim[2]) == l
    rand(l) .* (lim[2]-lim[1]) .+ lim[1]
end

function is_out(x, lim)
    any(x .>= lim[2]) || any(x .<= lim[1])
end

function run(env, x0, xlim, control, t0, tf; Δt=1.00,)
    x0 = State(env)(x0[1], x0[2])
    # oscillator = TwoDimensionalNonlinearOscillator()
    # env = Oscillator_ZOH_Input(oscillator, single_integrator)
    # x0 = State(env)(x0[1], x0[2])
    # p0 = zeros(1)  # initial parameter; would be meaningless
    simulator = Simulator(x0, apply_inputs(Dynamics!(env); u=(x, p, t) -> control(x, t));
                          Problem=:Discrete,
                          tf=tf)
    # df = solve(simulator)
    # sim
    df = DataFrame()
    for (i, t) in enumerate(t0:Δt:tf)
        state = simulator.integrator.u
        step_until!(simulator, t)
        if is_out(state, 2 .* xlim)
            break
        else
            push!(simulator, df)
        end
        # push!(simulator, df)
    end
    ts = df.time
    states = df.sol |> Map(datum -> datum.state) |> collect
    inputs = df.sol |> Map(datum -> datum.input) |> collect
    rewards = zip(states, inputs) |> MapSplat((state, input) -> FSimZoo.RunningCost(env)(state, input)) |> collect
    next_states = df.sol |> Map(datum -> datum.next_state) |> collect
    # data
    # xurx_nextData(states, inputs, rewards, next_states)
    txurx_nextData(ts, states, inputs, rewards, next_states)
end

# function q_learning!(normalised_approximator, xurx_next_data, ulim)
#     # Q learning
#     xurx_next_data_train, xurx_next_data_test = partitionTrainTest(xurx_next_data)
#     train_approximator!(normalised_approximator, xurx_next_data_train, xurx_next_data_test;
#                         loss=QLearningLoss(normalised_approximator; lim=ulim, verbose=true),  # limit while training
#                         epochs=30,
#                         opt=ADAM(1e-3),
#                         # opt=Flux.Optimiser(Flux.Optimise.ExpDecay(1e-3, 1e-1, 1000, 1e-5), ADAM()),
#                         )
#     # Supervised learning
#     # env = TwoDimensionalNonlinearDTSystem()
#     # xs, us = xurx_next_data.x, xurx_next_data.u
#     # fs = zip(xs, us) |> MapSplat((x, u) -> FSimZoo.OptimalQValue(env)(x, u)) |> collect
#     # xuf_data = xufData(xs, us, fs)
#     # xuf_data_train, xuf_data_test = partitionTrainTest(xuf_data)
#     # train_approximator!(normalised_approximator, xuf_data_train, xuf_data_test;
#     #                     loss=SupervisedLearningLoss(normalised_approximator),  # limit while training
#     #                     # loss=QLearningLoss(normalised_approximator; verbose=true),  # no limit while training
#     #                     epochs=30,
#     #                     opt=ADAM(1e-3),
#     #                     # opt=Flux.Optimiser(Flux.Optimise.ExpDecay(1e-3, 1e-1, 1000, 1e-5), ADAM()),
#     #                     )
# end

function finite_horizon_q_learning(txurx_next_data_t, ulim, terminal_value_func)
    # Q learning
    normalised_approximators = []
    for t in reverse(0:length(txurx_next_data_t)-2) # time index; reverse
        println("Training Q function for time $(t)... (total: $(length(txurx_next_data_t)-2) to 0)")
        data = txurx_next_data_t[begin+t]
        xs = data.x
        us = data.u
        rs = data.r
        x_nexts = data.x_next
        optvals = nothing
        if length(normalised_approximators) == 0
            optvals = x_nexts |> Map(x_next -> terminal_value_func(x_next)) |> collect
        else
            optvals = x_nexts |> Map(x_next -> solve!(normalised_approximators[end], x_next; lim=ulim).optval) |> tcollect
        end
        fs = zip(rs, optvals) |> MapSplat((r, optval) -> r+optval) |> collect
        xuf_data = xufData(xs, us, fs)
        normalised_approximator = generate_approximator(xuf_data)
        xuf_data_train, xuf_data_test = partitionTrainTest(xuf_data)
        train_approximator!(normalised_approximator, xuf_data_train, xuf_data_test;
                            loss=SupervisedLearningLoss(normalised_approximator),  # limit while training
                            epochs=100,
                            opt=ADAM(1e-3),
                           )
        push!(normalised_approximators, normalised_approximator)
    end
    normalised_approximators = reverse(normalised_approximators)
end

function generate_approximator(data)
    n, m, d = length(data.x[1]), length(data.u[1]), data.d
    # i_max = 20
    i_max = 50
    h_array = [64, 64, 64]
    T = 1e-1
    act = Flux.leakyrelu
    # pma = PMA(n, m, i_max, h_array, act)
    α_is = 1:i_max |> Map(i -> rand(n+m)) |> collect
    β_is = 1:i_max |> Map(i -> rand(1)) |> collect
    # normalised_approximator = NormalisedApproximator(LSE(α_is, β_is, T; n=n, m=m),
    #                                                  MinMaxNormaliser(data))  # meaningless normaliser
    normalised_approximator = NormalisedApproximator(PLSE(n, m, i_max, T, h_array, act),
                                                     MinMaxNormaliser(data))  # meaningless normaliser
end

function split_and_merge(data, t_index)
    data_vectorised = vectorise(data)
    data_vectorised_t = filter(datum -> datum.t == [t_index], data_vectorised)
    data_t = cat(data_vectorised_t)
end

function evaluate(data, terminal_value_func)
    total_reward = sum(data.r[1:end-1]) + terminal_value_func(data.x[end])
    evaluation = (; total_reward=total_reward)
end

function main(; seed=2021)
    Random.seed!(seed)
    n_scenario = 10_000
    env = TwoDimensionalNonlinearDTSystem()
    terminal_value_func = FSimZoo.OptimalValue(env)
    n, m = 2, 1
    t0, tf = 0, 5  # time horizon
    xlim = 1.0 .* (-1*ones(n), 1*ones(n))
    ulim = 1.0 .* (-1*ones(m), 1*ones(m))
    control_sample(state, t) = sample(ulim)
    x0s = 1:n_scenario |> Map(i -> sample(xlim)) |> collect
    @time data_scenarios = x0s |> Map(x0 -> run(env, x0, xlim, control_sample, t0, tf)) |> collect
    data = cat(data_scenarios)
    data_t = t0:tf |> Map(t -> split_and_merge(data, t)) |> collect  # e.g. data_t[begin+5].t will give you data with t=5
    @show "number of data = $(data.x |> length)"
    # training
    # normalised_approximators = generate_approximators(tf-t0, data_t)
    normalised_approximators = finite_horizon_q_learning(data_t, ulim, terminal_value_func)
    # control_trained(x, t) = [solve!(normalised_approximators[Int(begin+t)], x; lim=ulim).minimiser]
    control_trained(x, t) = t < tf ? [solve!(normalised_approximators[Int(begin+t)], x; lim=ulim).minimiser] : zeros(m)  # zero input at terminal time; meaningless
    x0_test = 0.50 .* [1, 1]
    data_optimal = run(env, x0_test, xlim, (x, t) -> FSimZoo.OptimalControl(env)(x), t0, tf)
    data_test = run(env, x0_test, xlim, control_trained, t0, tf)
    @show evaluate(data_optimal, terminal_value_func)
    @show evaluate(data_test, terminal_value_func)
    fig_x = plot()
    fig_u = plot()
    plot!(fig_x,
          hcat(data_optimal.t...)', hcat(data_optimal.x...)';
          ylim=(-1, 1),
          st=:scatter,
          markershape=:star5,
          color=[:blue :black],
          label=[L"x_{1}^{*}" L"x_{2}^{*}"],
         )
    plot!(fig_x,
          hcat(data_test.t...)', hcat(data_test.x...)';
          ylim=(-1, 1),
          st=:scatter,
          color=[:red :orange],
          label=[L"x_{1}" L"x_{2}"],
         )
    plot!(fig_u,
          hcat(data_optimal.t...)', hcat(data_optimal.u...)';
          ylim=(ulim[1][1], ulim[2][1]),
          st=:scatter,
          markershape=:star5,
          color=:blue,
          label=L"u^{*}",
         )
    plot!(fig_u,
          hcat(data_test.t...)', hcat(data_test.u...)';
          ylim=(ulim[1][1], ulim[2][1]),
          st=:scatter,
          color=:red,
          label=L"u",
         )
    fig_traj = plot(fig_x, fig_u; layout=(2, 1))
    dir_log = "figures/q_learning"
    savefig(fig_traj, joinpath(dir_log, "traj.png"))
    u_for_q = zeros(1)
    fig_q_true = plot(
                      xlim[1][1] : 0.01 : xlim[2][1],
                      xlim[1][2] : 0.01 : xlim[2][2],
                      (x1, x2) -> FSimZoo.OptimalQValue(env)(State(env)(x1, x2), u_for_q);
                      st=:surface,
                      zlim=(0, 10),
                     )
    fig_q = plot(
                 xlim[1][1] : 0.01 : xlim[2][1],
                 xlim[1][2] : 0.01 : xlim[2][2],
                 (x1, x2) -> normalised_approximators[end](State(env)(x1, x2), u_for_q)[1];
                 st=:surface,
                 zlim=(0, 10),
                )
    savefig(fig_q_true, joinpath(dir_log, "q_true.png"))
    savefig(fig_q, joinpath(dir_log, "q.png"))
end
