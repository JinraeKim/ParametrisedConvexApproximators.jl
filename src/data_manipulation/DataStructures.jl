abstract type AbstractDataStructure end

"""
Data structure for basic usage (e.g., Transducers.jl)
"""
struct xufData <: AbstractDataStructure
    x
    u
    f
    d  # no. of data
    function xufData(xs::Vector, us::Vector, fs::Vector)
        @assert length(xs) == length(us)
        @assert length(xs) == length(fs)
        d = length(xs)
        new(xs, us, fs, d)
    end
end


function Data_to_Flux(data::xufData)
    (; x=hcat(data.x...), u=hcat(data.u...), f=hcat(data.f...))  # NamedTuple
end

function Flux_to_Data(data::NamedTuple)
    xs = [data.x[:, i] for i in data.d]
    us = [data.u[:, i] for i in data.d]
    fs = [data.f[:, i] for i in data.d]
    xufData(xs, us, fs)
end
