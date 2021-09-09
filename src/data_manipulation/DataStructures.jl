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


"""
Data structure for Flux.jl
"""
struct xufFlux <: AbstractDataStructure
    x
    u
    f
    d::Int  # no. of data
    function xufFlux(x::Matrix, u::Matrix, f::Matrix)
        @assert size(x)[2] == size(u)[2]
        @assert size(x)[2] == size(f)[2]
        d = size(x)[2]
        new(x, u, f, d)
    end
end

function Data_to_Flux(data::xufData)
    xufFlux(hcat(data.x...), hcat(data.u...), hcat(data.f))
end

function Flux_to_Data(data::xufFlux)
    xs = [data.x[:, i] for i in data.d]
    us = [data.u[:, i] for i in data.d]
    fs = [data.f[:, i] for i in data.d]
    xufData(xs, us, fs)
end
