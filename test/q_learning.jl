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

function run(x0, xlim;
        Δt=0.003, ulim=(-5*ones(1), 5*ones(1)),
    )
    # env
    oscillator = TwoDimensionalNonlinearOscillator()
    env = Oscillator_ZOH_Input(oscillator)
    x0 = State(env)(x0[1], x0[2])
    t0 = 0.0
    tf = 3.0
    p0 = zeros(1)  # initial parameter; would be meaningless
    simulator = Simulator(x0, Dynamics!(env), p0; tf=tf)
    # sim
    df = DataFrame()
    optimal_control = FSimZoo.OptimalControl(env.oscillator)
    for (i, t) in enumerate(t0:Δt:tf)
        state = simulator.integrator.u
        if (i-1) % 1 == 0  # update input period
            simulator.integrator.p = sample(ulim)
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

function main(; seed=2021)
    Random.seed!(seed)
    n_scenario = 100
    xlim = (-1*ones(2), 1*ones(2))
    x0s = 1:n_scenario |> Map(i -> sample(xlim)) |> collect
    @time data_scenarios = x0s |> Map(x0 -> run(x0, xlim)) |> tcollect
    data = cat(data_scenarios)
    @show data.x
    nothing
end
