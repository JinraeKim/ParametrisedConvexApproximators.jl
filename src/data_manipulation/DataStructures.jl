abstract type AbstractDataStructure end

"""
Data structure for basic usage (e.g., Transducers.jl) with keys `x`, `u`, and `f`.
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


"""
Data structure for basic usage (e.g., Transducers.jl) with keys `x`, `u`, `r`, and `x_next`.
"""
struct xurx_nextData <: AbstractDataStructure
    x
    u
    r
    x_next
    d  # no. of data
    function xurx_nextData(xs::Vector, us::Vector, rs::Vector, x_nexts::Vector)
        d = length(xs)
        @assert length(us) == d
        @assert length(rs) == d
        @assert length(x_nexts) == d
        new(xs, us, rs, x_nexts, d)
    end
end

function Data_to_NamedTuple(data::xufData)
    (; x=hcat(data.x...), u=hcat(data.u...), f=hcat(data.f...))  # NamedTuple
end

function Data_to_NamedTuple(data::xuData)
    (; x=hcat(data.x...), u=hcat(data.u...))  # NamedTuple
end

function Data_to_NamedTuple(data::xurx_nextData)
    (; x=hcat(data.x...), u=hcat(data.u...), r=hcat(data.r...), x_next=hcat(data.x_next...))  # NamedTuple
end

function Base.cat(data_vector::Vector{xurx_nextData})
    x_big = cat((data_vector |> Map(data -> data.x) |> collect)...; dims=1)
    u_big = cat((data_vector |> Map(data -> data.u) |> collect)...; dims=1)
    r_big = cat((data_vector |> Map(data -> data.r) |> collect)...; dims=1)
    x_next_big = cat((data_vector |> Map(data -> data.x_next) |> collect)...; dims=1)
    xurx_nextData(x_big, u_big, r_big, x_next_big)
end
