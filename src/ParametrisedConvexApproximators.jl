module ParametrisedConvexApproximators

using Flux
using UnPack
using Convex, Mosek, MosekTools
using Optim
using Transducers


# approximators
export AbstractApproximator, FNN
export ParametrisedConvexApproximator, PLSE
export optimise


# approximators
include("approximators/approximators.jl")
include("optimise.jl")


end  # module
