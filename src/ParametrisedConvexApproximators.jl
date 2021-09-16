module ParametrisedConvexApproximators

# using Debugger  # tmp
using Flux
using Random
using Transducers
using Convex

# data structure
export xufData
export partitionTrainTest
# approximators
export pMA
# training
export train_approximator!


include("approximators/approximators.jl")
include("data_manipulation/data_manipulation.jl")
include("training/training.jl")
include("visualisation/visualisation.jl")


end  # module
