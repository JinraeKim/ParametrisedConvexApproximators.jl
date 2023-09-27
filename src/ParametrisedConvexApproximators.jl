module ParametrisedConvexApproximators

using Flux
using Convex
using ECOS
using Optim
using Random
using ForwardDiff  # mostly for DLSE
using ImplicitDifferentiation  # for the differentiation of the minimiser
using ComponentArrays
using Distributed  # for pmap
using Printf
using ProgressMeter
using AccessorsExtra


# approximators
export AbstractApproximator, FNN
export ConvexApproximator, MA, LSE
export ParametrisedConvexApproximator, PMA, PLSE, PLSEplus
export PICNN, project_nonnegative!
export DifferenceOfConvexApproximator, DLSE
export EPLSE
export minimise, number_of_parameters
export MaxAbsNormalisedApproximator
# data manipulation
export split_data2, split_data3
# dataset
export generate_dataset, DecisionMakingDataset
export example_target_function
# trainer
export SupervisedLearningTrainer
export get_loss


# approximators
include("dataset/dataset.jl")
include("approximators/approximators.jl")
include("minimise.jl")
include("trainer/trainer.jl")


end  # module
