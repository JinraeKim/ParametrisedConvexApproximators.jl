abstract type AbstractNormaliser end


struct StandardNormalDistributionNormaliser <: AbstractNormaliser
    mean_nt
    std_nt
    function StandardNormalDistributionNormaliser(data::xufData)
        @unpack x, u, f = data
        mean_nt = (; x=mean(x), u=mean(u), f=mean(f))
        std_nt = (; x=std(x), u=std(u), f=std(f))
        new(mean_nt, std_nt)
    end
end

"""
z ∈ ℝ^l or ℝ^(l×d)
"""
function normalise(normaliser::StandardNormalDistributionNormaliser, z::Convex.AbstractExpr, name)
    # TODO: unify this method to normalise(normaliser::StandardNormalDistributionNormaliser, z, name)
    @unpack mean_nt, std_nt = normaliser
    (z - getproperty(mean_nt, name)) ./ getproperty(std_nt, name)
end

function normalise(normaliser::StandardNormalDistributionNormaliser, z, name)
    @unpack mean_nt, std_nt = normaliser
    (z .- getproperty(mean_nt, name)) ./ getproperty(std_nt, name)
end

function unnormalise(normaliser::StandardNormalDistributionNormaliser, z, name)
    @unpack mean_nt, std_nt = normaliser
    (z .* getproperty(std_nt, name)) .+ getproperty(mean_nt, name)
end


"""
DO NOT USE `min.(x...)` instead of `minimum(_xs, dims=2)[:]`; horribly slow.
"""
struct MinMaxNormaliser <: AbstractNormaliser
    min_nt
    max_nt
    function MinMaxNormaliser(data::xufData)
        @unpack x, u, f = data
        _xs = hcat(x...)
        _us = hcat(u...)
        _fs = hcat(f...)
        min_nt = (; x=minimum(_xs, dims=2)[:], u=minimum(_us, dims=2)[:], f=minimum(_fs, dims=2)[:])  # vectorise
        max_nt = (; x=maximum(_xs, dims=2)[:], u=maximum(_us, dims=2)[:], f=maximum(_fs, dims=2)[:])  # vectorise
        new(min_nt, max_nt)
    end
end

"""
z ∈ ℝ^l or ℝ^(l×d)

each element of z or z[:, i] (if z is Matrix) is mapped from [z_min, z_max] to [0, 1]
"""
function normalise(normaliser::MinMaxNormaliser, z::Convex.AbstractExpr, name)
    @unpack min_nt, max_nt = normaliser
    (z - getproperty(min_nt, name)) ./ (getproperty(max_nt, name) - getproperty(min_nt, name))
end

function normalise(normaliser::MinMaxNormaliser, z, name)
    @unpack min_nt, max_nt = normaliser
    (z .- getproperty(min_nt, name)) ./ (getproperty(max_nt, name) - getproperty(min_nt, name))
end

"""
z ∈ ℝ^l or ℝ^(l×d)

each element of z or z[:, i] (if z is Matrix) is mapped from [0, 1] to [z_min, z_max]
"""
function unnormalise(normaliser::MinMaxNormaliser, z, name)
    @unpack min_nt, max_nt = normaliser
    getproperty(min_nt, name) .+ (z .* (getproperty(max_nt, name) - getproperty(min_nt, name)))
end
