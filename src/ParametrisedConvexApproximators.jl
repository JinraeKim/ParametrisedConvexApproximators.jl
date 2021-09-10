module ParametrisedConvexApproximators

using Debugger  # tmp
using Flux
using Random
using Transducers


include("approximators/approximators.jl")
include("data_manipulation/data_manipulation.jl")
include("training/training.jl")
include("visualisation/visualisation.jl")


end  # module
