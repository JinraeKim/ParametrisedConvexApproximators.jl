abstract type AbstractNormaliser end


struct Normaliser <: AbstractNormaliser
    mean_nt
    std_nt
    function Normaliser(data::xufData)
        @unpack x, u, f = data
        mean_nt = (; x=mean(x), u=mean(u), f=mean(f))
        std_nt = (; x=std(x), u=std(u), f=std(f))
        new(mean_nt, std_nt)
    end
end

"""
z ∈ ℝ^l or ℝ^(l×d)
"""
function normalise(normaliser::Normaliser, z::Convex.AbstractExpr, name)
    # z
    # TODO: unify this method to normalise(normaliser::Normaliser, z, name)
    @unpack mean_nt, std_nt = normaliser
    (z - getproperty(mean_nt, name)) / getproperty(std_nt, name)
end

function normalise(normaliser::Normaliser, z, name)
    # z
    @unpack mean_nt, std_nt = normaliser
    (z .- getproperty(mean_nt, name)) ./ getproperty(std_nt, name)
end

function unnormalise(normaliser::Normaliser, z, name)
    z
    @unpack mean_nt, std_nt = normaliser
    (z .* getproperty(std_nt, name)) .+ getproperty(mean_nt, name)
end
