using RBM

binary_draw(p::Array{Float64})                                  = map(x->x?1.0:0.0, p .> rand(size(p)))
sigm(p::Vector{Float64})                                        = 1. ./ (1. + exp(-p))

down(rbm::RBM_t,        z::Vector{Float64})                     = sigm(rbm.b + rbm.W' * z)
binary_down(rbm::RBM_t, z::Vector{Float64})                     = binary_draw(down(rbm, z))

up(rbm::RBM_t,          y::Vector{Float64}, x::Vector{Float64}) = sigm(rbm.c + rbm.V * y + rbm.W * x)
binary_up(rbm::RBM_t,   y::Vector{Float64}, x::Vector{Float64}) = binary_draw(up(rbm, y, x))

sigm(p::Matrix{Float64})                                        = 1. ./ (1. + exp(-p))
down(rbm::RBM_t,        z::Matrix{Float64})                     = sigm(repmat(rbm.b, 1, size(z)[2]) + rbm.W' * z)
binary_down(rbm::RBM_t, z::Matrix{Float64})                     = binary_draw(down(rbm, z))

up(rbm::RBM_t,          y::Matrix{Float64}, x::Matrix{Float64}) = sigm(repmat(rbm.c, 1, size(y)[2]) + rbm.V * y + rbm.W * x)
binary_up(rbm::RBM_t,   y::Matrix{Float64}, x::Matrix{Float64}) = binary_draw(up(rbm, y, x))
