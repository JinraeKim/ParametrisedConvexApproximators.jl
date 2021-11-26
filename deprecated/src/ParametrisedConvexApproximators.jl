module ParametrisedConvexApproximators

# using Debugger  # tmp
using Flux
using Zygote
using Random
using Transducers
using Convex
using Mosek, MosekTools
using Optim
using UnPack
using RecipesBase: AbstractPlot, plot!
using Statistics: mean, std
using BenchmarkTools

# data structure
export xufData, xurx_nextData, txurx_nextData
export partitionTrainTest, vectorise
## normaliser
export IdentityNormaliser, MinMaxNormaliser, StandardNormalDistributionNormaliser
# approximators
export ParametrisedConvexApproximator, ConvexApproximator
export FNN
export MA, LSE
export PMA, PLSE
export NormalisedApproximator
export solve!
export number_of_params
# training
export train_approximator!
export plot_approx!
export SupervisedLearningLoss, QLearningLoss


include("data_manipulation/data_manipulation.jl")
include("approximators/approximators.jl")
include("training/training.jl")
include("visualisation/visualisation.jl")


end  # module
