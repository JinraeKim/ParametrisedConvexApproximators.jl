"""
    PICNN

Partially input convex neural network [1].
# References
[1] B. Amos, L. Xu, and J. Z. Kolter, “Input Convex Neural Networks,” in 34th International Conference on Machine Learning, ICML 2017, 2017, vol. 1, pp. 192–206.
# Variables
(!!NOTICE!!) The variables in this code probably follow the notation of the original paper as follows:
`x` and `y` denote non-convex and convex inputs, respectively.
`g̃` and `g` denote activation functions of `x`- and `y`- paths, respectively.
Furthermore, `u` and `z` denote `x`- and `y`-path variables, respectively.
"""
struct PICNN <: ParametrisedConvexApproximator
    n::Int
    m::Int
    NN::Flux.Chain
    function PICNN(n::Int, m::Int, u_array::Vector{Int}, z_array::Vector{Int}, g, g̃)
        new(n, m, make_PICNN(n, m, u_array, z_array, g, g̃))
    end
end
# PICNN(n::Int, m::Int, u_array, z_array; g=Flux.identity, g̃=Flux.identity) = PICNN(n, m, make_PICNN(n, m, u_array, z_array, g, g̃))

# """
# # Notes
# - The output activation function of y-path, i.e., g, is identity.
# """
function make_PICNN(n, m, u_array, z_array, g, g̃)
    if g != Flux.relu && g != Flux.leakyrelu
        error("Unsupported activation function; it should be convex and non-decreasing")
    end
    if length(u_array) != length(z_array) + 1
        error("Invalid length of arrays")
    end
    PICNN_Layers = []
    _u_array = [n, u_array...]
    _z_array = [1, z_array..., 1]
    for i in 1:length(_u_array)-1
        _g = i == length(_u_array)-1 ? Flux.identity : g
        push!(PICNN_Layers, PICNN_Layer(_u_array[i], _u_array[i+1], _z_array[i], _z_array[i+1], m, _g, g̃))
    end
    Chain(Init_PICNN_Layer, PICNN_Layers..., Final_PICNN_Layer)
end

# forward
function (nn::PICNN)(x, y)
    res = nn.NN((x, y))
    if typeof(y) <: Convex.AbstractExpr
        res = [res]  # to make it a vector whose length = 1; consistent with typeof(y) <: AbstractArray
    end
    return res
end

## PICNN_Layer
struct PICNN_Layer
    # info
    n_in  # not trainable; will not be tracked by Flux, automatically 
    n_out  # not trainable; will not be tracked by Flux, automatically 
    # x-path
    W̃  #  params(m)[1]
    b̃  #  params(m)[2]
    g̃  # not trainable; will not be tracked by Flux, automatically 
    # y-path
    Wz  #  params(m)[3] >= 0 (element-wise)
    Wzu  #  params(m)[4]
    bz  #  params(m)[5]
    Wy  #  params(m)[6]
    Wyu  #  params(m)[7]
    by  #  params(m)[8]
    Wu  #  params(m)[9]
    b  #  params(m)[10]
    g  # not trainable; will not be tracked by Flux, automatically 
end

function PICNN_Layer(uin::Int, uout::Int, zin::Int, zout::Int, y::Int, g=Flux.identity, g̃=Flux.identity;
        initW = Flux.glorot_uniform, initb = zeros  # default initialisation method
    )
    layer = PICNN_Layer(
                       uin, y,  # in & out
                       initW(uout, uin), initb(uout), g̃,  # W̃, b̃, g̃ (x-path)
                       max.(initW(zout, zin), 0.0), initW(zin, uin), initb(zin),  # Wz > 0, Wzu, bz
                       initW(zout, y), initW(y, uin), initb(y),  # Wy, Wyu, by
                       initW(zout, uin), initb(zout),  # Wu, b
                       g,  # g (y-path activation)
                      )
end
Flux.@functor PICNN_Layer  # make "struct" compatible with Flux

function Flux.relu(x::Convex.AbstractExpr)
    Convex.pos(x)
end
function Flux.leakyrelu(x::Convex.AbstractExpr)
    Convex.max(x, 0.1*x)
end
function (nn::PICNN_Layer)(input)
    u, z, y = input
    # network params
    W̃, b̃, g̃ = nn.W̃, nn.b̃, nn.g̃
    Wz, Wzu, bz = nn.Wz, nn.Wzu, nn.bz
    Wy, Wyu, by = nn.Wy, nn.Wyu, nn.by
    Wu, b = nn.Wu, nn.b
    g = nn.g
    if typeof(y) <: Convex.AbstractExpr
        u_next = g̃.(W̃*u .+ b̃)
        z_next = g(
            Wz * dot(*)(z, max.(Wzu*u .+ bz, 0.0))  # dot(*) is Hadamard product in Convex
            + Wy * dot(*)(y, (Wyu*u .+ by))
            + (Wu * u .+ b)
        )  # broadcasting is not supported by Convex.jl
    else
        u_next = g̃.(W̃*u .+ b̃)
        z_next = g.(
            Wz * dot(*)(z, max.(Wzu*u .+ bz, 0.0))  # dot(*) is Hadamard product in Convex
            + Wy * dot(*)(y, (Wyu*u .+ by))
            + (Wu * u .+ b)
        )
    end
    return u_next, z_next, y
end

function Init_PICNN_Layer(input)
    x, y = input
    return x, Float32(0), y  # not `x, zeros(1), y` for CUDA
end

function Final_PICNN_Layer(input)
    u, z, y = input
    return z
end

function project_nonnegative!(ps; ϵ=0.0)
    ps .-= ps .* (ps .< ϵ)
end

function Flux.params(nn::PICNN)
    Flux.params(nn.NN)
end
