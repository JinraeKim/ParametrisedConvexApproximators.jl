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
Data structure for basic usage (e.g., Transducers.jl)
"""
struct xuData <: AbstractDataStructure
    x
    u
    d  # no. of data
    function xuData(xs::Vector, us::Vector)
        @assert length(xs) == length(us)
        d = length(xs)
        new(xs, us, d)
    end
end

function Data_to_NamedTuple(data::xufData)
    (; x=hcat(data.x...), u=hcat(data.u...), f=hcat(data.f...))  # NamedTuple
end

function Data_to_NamedTuple(data::xuData)
    (; x=hcat(data.x...), u=hcat(data.u...))  # NamedTuple
end
