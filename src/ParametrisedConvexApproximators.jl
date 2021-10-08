module ParametrisedConvexApproximators

# using Debugger  # tmp
using Flux
using Random
using Transducers
using Convex
# import Convex
using Mosek, MosekTools
using UnPack
using RecipesBase: AbstractPlot, plot!
using Statistics: mean, std

# data structure
export xufData
export partitionTrainTest
export MinMaxNormaliser, StandardNormalDistributionNormaliser
# approximators
export NormalisedApproximator
export ParametrisedConvexApproximator, ConvexApproximator
export FNN
export MA, LSE
export PMA, PLSE
# training
export train_approximator!
export plot_approx!
export solve!


include("data_manipulation/data_manipulation.jl")
include("approximators/approximators.jl")
include("training/training.jl")
include("visualisation/visualisation.jl")


end  # module
