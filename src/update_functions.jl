using RBM
b2d(x)                                                        = (x==true)?1.0:0.0
binary_draw(p::Array{Float64})                                = map(b2d, p .> rand(size(p)))
sigm(p::Array{Float64})                                       = 1. ./ (1. + exp(-p))

down(rbm::RBM_t,        z::Array{Float64})                    = transpose(sigm(repmat(rbm.b, 1, size(z)[1]) + rbm.W' * z'))
binary_down(rbm::RBM_t, z::Array{Float64})                    = binary_draw(down(rbm, z))

up(rbm::RBM_t,          y::Array{Float64}, x::Array{Float64}) = transpose(sigm(repmat(rbm.c, 1, size(y)[1]) + rbm.V * y' + rbm.W * x'))
binary_up(rbm::RBM_t,   y::Array{Float64}, x::Array{Float64}) = binary_draw(up(rbm, y, x))
