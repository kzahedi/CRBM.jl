module CRBM

using RBM
using ProgressMeter
using PyPlot
using Shannon

export crbm_control_sample
export crbm_binary_train!

#export from RBM package, so that you don't have to import both
export RBM_t
export CRBM_cfg_t, crbm_create_config
export rbm_copy
export rbm_create
export rbm_write, rbm_read
export rbm_init_weights_random!
export rbm_init_visible_bias!
export rbm_init_output_bias_random!, rbm_init_hidden_bias_random!

export binarise_matrix, i2b, iv2b
export unbinarise_matrix, b2i, b2iv, bv2dv
export bin_value, bin_matrix, bin_vector
export unbin_value, unbin_matrix, unbin_vector
export binary_draw
export up, down, binary_up, binary_down

type CRBM_cfg_t
  use_progress_meter::Bool
  use_pyplot::Bool
end

function crbm_create_config()
  return CRBM_cfg_t(true, true)
end

# 
# start - covered by test cases
#
i2b(v::Int64, n::Int64) = [(((1 << (n-i)) & v)>0)?1.0:0.0 for i=1:n]
b2i(v::Vector{Float64}) = int64(foldl(+, [(v[i]>0)?(1<<(length(v)-i)):0 for i=1:length(v)]))

iv2b(v::Vector{Int64},   n::Int64) = foldl(vcat, [i2b(u ,n) for u in v])
b2iv(v::Vector{Float64}, n::Int64) = foldl(vcat, [b2i(v[i:i+(n-1)]) for i=1:n:length(v)])

bv2dv(v::Vector{Float64}, bins::Int64; mode="centre") = map(x->unbin_value(int64(x+1), bins, -1.0, 1.0, mode=mode), b2iv(v, int(ceil(log2(bins)))))

function binarise_matrix(A::Matrix{Float64}, bins::Int64)
  N = int(ceil(log2(bins)))
  B = zeros(size(A)[1], size(A)[2]* N)
  C = bin_matrix(A, -1.0, 1.0, bins) .- 1
  for row_index = 1:size(A)[1]
    B[row_index,:] = iv2b(squeeze(C[row_index,:],1), N)
  end
  B
end

function unbinarise_matrix(A::Matrix{Float64}, bins::Int64; mode="centre")
  N = int(ceil(log2(bins)))
  w = size(A)[1]
  v = int(size(A)[2] / N)
  B = zeros(w, v)
  for row_index = 1:w
    B[row_index,:] = b2iv(squeeze(A[row_index,:],1), N)
  end
  map(x->unbin_value(int64(x+1), bins, -1.0, 1.0, mode=mode), B)
end

binary_draw(p::Vector{Float64}) = map(x->x?1.0:0.0, p .> rand(size(p)))
binary_draw(p::Matrix{Float64}) = map(x->x?1.0:0.0, p .> rand(size(p)))

#
# end   - covered by test cases
# 

# start - tested but not with unit test
sigm(p::Vector{Float64})                                        = 1. ./ (1. + exp(-p))
down(rbm::RBM_t,        z::Vector{Float64})                     = sigm(rbm.b + rbm.W' * z)
up(rbm::RBM_t,          y::Vector{Float64}, x::Vector{Float64}) = sigm(rbm.c + rbm.V * y + rbm.W * x)
binary_up(rbm::RBM_t,   y::Vector{Float64}, x::Vector{Float64}) = binary_draw(up(rbm, y, x))
binary_down(rbm::RBM_t, z::Vector{Float64})                     = binary_draw(down(rbm, z))

sigm(p::Matrix{Float64})                                        = 1. ./ (1. + exp(-p))
down(rbm::RBM_t,        z::Matrix{Float64})                     = sigm(repmat(rbm.b, 1, size(z)[2]) + rbm.W' * z)
up(rbm::RBM_t,          y::Matrix{Float64}, x::Matrix{Float64}) = sigm(repmat(rbm.c, 1, size(y)[2]) + rbm.V * y + rbm.W * x)
binary_up(rbm::RBM_t,   y::Matrix{Float64}, x::Matrix{Float64}) = binary_draw(up(rbm, y, x))
binary_down(rbm::RBM_t, z::Matrix{Float64})                     = binary_draw(down(rbm, z))
# end   - tested but not with unit test


function crbm_learn_sampling(rbm::RBM_t, y::Vector{Float64}, X::Vector{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = binary_down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end
  X = binary_down(rbm, Z)
  Z = binary_up(rbm, y, X)
  return X,Z
end

function crbm_learn_sampling(rbm::RBM_t, y::Matrix{Float64}, X::Matrix{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = binary_down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end
  X = binary_down(rbm, Z)
  Z = binary_up(rbm, y, X)
  return X,Z
end

function crbm_control_sample(rbm::RBM_t, y::Vector{Float64}, X::Vector{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = binary_down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end
  X = binary_down(rbm, Z)
  return X
end


function crbm_binary_train!(cfg::CRBM_cfg_t, rbm::RBM_t, S::Matrix{Float64}, A::Matrix{Float64})
  @assert (rbm.dropout >= 0.0 && rbm.dropout <= 1.0) "Dropout must be in [0,1]"
  # TODO more assert statements needed
  N  = ceil(log2(rbm.bins))
  binary_s_matrix = binarise_matrix(S, rbm.bins)
  binary_a_matrix = binarise_matrix(A, rbm.bins)
  ns = size(binary_s_matrix[1,:])[1]

  if maximum(rbm.W) == 0.0 && minimum(rbm.W) == 0.0 && maximum(rbm.V) == 0.0 && minimum(rbm.V) == 0.0
    println("Initialising W, V, and c.")
    rbm_init_weights_random!(rbm)
    rbm.c = zeros(rbm.m)
  end

  rbm_init_visible_bias!(rbm, convert(Array{Int64},binary_s_matrix))

  binary_s_matrix = transpose(binary_s_matrix)
  binary_a_matrix = transpose(binary_a_matrix)

  println(size(binary_s_matrix))
  println(size(binary_a_matrix))

  if cfg.use_progress_meter == true
    pm = Progress(rbm.numepochs, 1)
  end
  for t=1:rbm.numepochs
    # extract data batch for current epoch
    m     = size(binary_s_matrix)[2] - rbm.batchsize
    start = int(1 + floor(rand() * m)) # 1 to m
    r     = [start:start+rbm.batchsize-1]
    s     = binary_s_matrix[:,r] # because it is transposed
    a     = binary_a_matrix[:,r] # because it is transposed

    # generate hidden states given the data
    z = binary_up(rbm, s, a)

    # generate random outputs to start sampler
    (A, Z) = crbm_learn_sampling(rbm, s, a)

    Eb  = transpose(mean(a,2) - mean(A,2))
    Ec  = transpose(mean(z,2) - mean(Z,2))
    EW  = (z * a' - Z * A')/ns
    EV  = (z * s' - Z * s')/ns

    Eb = squeeze(Eb,1)
    Ec = squeeze(Ec,1)

    rbm.b = rbm.b + rbm.alpha * Eb
    rbm.c = rbm.c + rbm.alpha * Ec
    rbm.W = rbm.W + rbm.alpha * EW
    rbm.V = rbm.V + rbm.alpha * EV

    if rbm.momentum > 0.0
      f = rbm.alpha * rbm.momentum
      rbm.b = rbm.b + f * rbm.vb
      rbm.c = rbm.c + f * rbm.vc
      rbm.W = rbm.W + f * rbm.vW
      rbm.V = rbm.V + f * rbm.vV
    end

    if rbm.weightcost > 0.0 # using L2
      f     = (1 - rbm.weightcost)
      rbm.W = rbm.W * f
      rbm.V = rbm.V * f
    end

    rbm.vb = Eb
    rbm.vc = Ec
    rbm.vW = EW
    rbm.vV = EV

    if cfg.use_pyplot == true
      if t % 500 == 0 || t == 0
        clf()
        subplot(121)
        colorbar(imshow(rbm.W))
        subplot(122)
        colorbar(imshow(rbm.V))
      end
    end

    next!(pm)
  end # training iteration
end

end
