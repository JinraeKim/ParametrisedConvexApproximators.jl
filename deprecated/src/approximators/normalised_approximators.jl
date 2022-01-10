struct NormalisedApproximator <: AbstractApproximator
    approximator::AbstractApproximator
    normaliser::AbstractNormaliser
end

function Base.show(io::IO, normalised_approximator::NormalisedApproximator)
    print(io,
          typeof(normalised_approximator.approximator),
          " (with normaliser ", 
          typeof(normalised_approximator.normaliser),
          ")", 
         )
end

"""
x ∈ ℝ^n or ℝ^(n×d)
u ∈ ℝ^m or ℝ^(m×d)
"""
# function (normalised_approximator::NormalisedApproximator)(x, u)
#     @unpack approximator, normaliser = normalised_approximator
#     x_normal = normalise(normaliser, x, :x)
#     u_normal = normalise(normaliser, u, :u)
#     f = approximator(x_normal, u_normal)
# end

# function (normalised_approximator::NormalisedApproximator)(x, u)
function (normalised_approximator::NormalisedApproximator)(x, u; output_normalisation=false)
    @unpack approximator, normaliser = normalised_approximator
    x_normal = normalise(normaliser, x, :x)
    u_normal = normalise(normaliser, u, :u)
    f_normal = approximator(x_normal, u_normal)
    f = f_normal
    # if output_normalisation
    #     f = f_normal
    # else
    #     f = unnormalise(normaliser, f_normal, :f)
    # end
    # f
end
