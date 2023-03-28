module ParametrisedConvexApproximators

using Flux
using Convex
using ECOS
using Optim
using Transducers  # mostly for parallel optimization
using Random
using ForwardDiff  # mostly for DLSE


# approximators
export AbstractApproximator, FNN
export ConvexApproximator, MA, LSE
export ParametrisedConvexApproximator, PMA, PLSE
export PICNN, project_nonnegative!
export DifferenceOfConvexApproximator, DLSE
export optimise, number_of_parameters
export MaxAbsNormalisedApproximator
# data manipulation
export split_data2, split_data3
# dataset
export SimpleDataset
# trainer
export SupervisedLearningTrainer
export get_loss


# approximators
include("dataset/dataset.jl")
include("approximators/approximators.jl")
include("optimise.jl")
include("trainer/trainer.jl")


end  # module
