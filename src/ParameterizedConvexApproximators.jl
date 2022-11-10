module ParameterizedConvexApproximators

using Flux
using Convex
using SCS  # 211207; currently not compatible with apple silicon
using Optim
using Transducers
using Random


# approximators
export AbstractApproximator, FNN
export ConvexApproximator, MA, LSE
export ParameterizedConvexApproximator, PMA, PLSE
export PICNN, project_nonnegative!
export optimize, number_of_parameters
# data manipulation
export split_data2, split_data3
# dataset
export SimpleDataset
# trainer
export SupervisedLearningTrainer
export get_loss


# approximators
include("approximators/approximators.jl")
include("optimize.jl")
include("data_manipulation/data_manipulation.jl")
include("dataset/dataset.jl")
include("trainer/trainer.jl")


end  # module
