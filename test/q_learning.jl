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


struct Oscillator_ZOH_Input <: AbstractEnv
    oscillator
end

function FS.State(env::Oscillator_ZOH_Input)
    @unpack oscillator = env
    State(oscillator)
end

function FS.Dynamics!(env::Oscillator_ZOH_Input)
    @unpack oscillator = env
    @Loggable function dynamics!(dx, x, input, t)
        @nested_log Dynamics!(oscillator)(dx, x, nothing, t; u=input)
    end
end

function sample(lim)
    @assert length(lim) == 2
    l = length(lim[1])
    @assert length(lim[2]) == l
    rand(l) .* (lim[2]-lim[1]) .+ lim[1]
end

function is_out(x, lim)
    any(x .>= lim[2]) || any(x .<= lim[1])
end

function run(x0, xlim, control;
        Δt=0.003,
        t0=0.0,
        tf=10.0,
    )
    # env
    oscillator = TwoDimensionalNonlinearOscillator()
    env = Oscillator_ZOH_Input(oscillator)
    x0 = State(env)(x0[1], x0[2])
    p0 = zeros(1)  # initial parameter; would be meaningless
    simulator = Simulator(x0, Dynamics!(env), p0; tf=tf)
    # sim
    df = DataFrame()
    # optimal_control = FSimZoo.OptimalControl(env.oscillator)
    for (i, t) in enumerate(t0:Δt:tf)
        state = simulator.integrator.u
        if (i-1) % 1 == 0  # update input period
            simulator.integrator.p = control(state)
        end
        step_until!(simulator, t)
        if is_out(state, xlim)
            break
        else
            push!(simulator, df)
        end
    end
    _states = df.sol |> Map(datum -> datum.state) |> collect
    _inputs = df.sol |> Map(datum -> datum.input) |> collect
    # data
    states = _states[1:end-1]
    inputs = _inputs[1:end-1]
    rewards = zip(states, inputs) |> MapSplat((x, u) -> FSimZoo.RunningCost(env.oscillator)(x, u)*Δt) |> collect  # approximate reward
    next_states = _states[2:end]
    xurx_nextData(states, inputs, rewards, next_states)
end

function q_learning!(normalised_approximator, xurx_next_data, ulim)
    xurx_next_data_train, xurx_next_data_test = partitionTrainTest(xurx_next_data)
    train_approximator!(normalised_approximator, xurx_next_data_train, xurx_next_data_test;
                        loss=QLearningLoss(normalised_approximator; lim=ulim),
                        epochs=300,
                        opt=ADAM(5e-5),
                        )
end

function generate_approximator(xuf_data)
    n, m, d = length(xuf_data.x[1]), length(xuf_data.u[1]), xuf_data.d
    i_max = 20
    h_array = [64, 64, 64]
    T = 1e-1
    act = Flux.leakyrelu
    plse = PLSE(n, m, i_max, T, h_array, act)
    NormalisedApproximator(plse,
                           MinMaxNormaliser(xuf_data))
end

function main(; seed=2021)
    Random.seed!(seed)
    n_scenario = 10
    xlim = (-2*ones(2), 2*ones(2))
    ulim=(-6*ones(1), 6*ones(1))
    control_sample(state) = sample(ulim)
    x0s = 1:n_scenario |> Map(i -> sample(xlim)) |> collect
    @time data_scenarios = x0s |> Map(x0 -> run(x0, xlim, control_sample)) |> collect
    data = cat(data_scenarios)
    @show "number of data = $(data.x |> length)"
    # training
    normalised_approximator = generate_approximator(data)
    q_learning!(normalised_approximator, data, ulim)
    # TODO: evaluation
    control_trained(x) = [solve!(normalised_approximator, x; lim=ulim).minimiser]
    x0_test = [0.5, 0.5]
    @show control_trained(x0_test)
    @show -3*x0_test[2]
    data_test = run(x0_test, xlim, control_trained)
    fig = plot(1:length(data_test.x), hcat(data_test.x...)')
    @show hcat(data_test.x...)' |> size
    display(fig)
end
