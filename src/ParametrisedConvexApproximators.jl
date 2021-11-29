module ParametrisedConvexApproximators

using Flux
using UnPack
using Convex, Mosek, MosekTools
using Optim
using Transducers


# approximators
export AbstractApproximator, FNN
export ConvexApproximator, MA, LSE
export ParametrisedConvexApproximator, PMA, PLSE
export optimise


# approximators
include("approximators/approximators.jl")
include("optimise.jl")


end  # module
