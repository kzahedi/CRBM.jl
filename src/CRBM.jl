module CRBM

include("conversion_functions.jl")
include("update_functions.jl")

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

export binarise_matrix, i2b, iv2b, binarise_vector
export unbinarise_matrix, b2i, b2iv, bv2dv
export bin_value, bin_matrix, bin_vector
export unbin_value, unbin_matrix, unbin_vector
export binary_draw
export up, down, binary_up, binary_down

type CRBM_cfg_t
  use_progress_meter::Bool
  use_pyplot::Bool
  plot_steps::Int64
  batchmode::String
end

function crbm_create_config()
  return CRBM_cfg_t(true, true, 100, "random")
end

function crbm_learn_sampling(rbm::RBM_t, y::Array{Float64}, X::Array{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = binary_down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end
  X = binary_down(rbm, Z)
  Z = binary_up(rbm, y, X)
  return X,Z
end

function crbm_control_sample(rbm::RBM_t, y::Array{Float64}, X::Array{Float64})
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
  println("binarising sensor data")
  binary_s_matrix = binarise_matrix(S, rbm.bins)
  println("binarising actuator data")
  binary_a_matrix = binarise_matrix(A, rbm.bins)
  ds = size(S)
  ns = ds[1]

  if maximum(rbm.W) == 0.0 && minimum(rbm.W) == 0.0 && maximum(rbm.V) == 0.0 && minimum(rbm.V) == 0.0
    println("Initialising W, V, and c.")
    rbm_init_weights_random!(rbm)
    rbm.c = zeros(rbm.m)
  end

  println("Initialising visible bias")
  rbm_init_visible_bias!(rbm, convert(Array{Int64},binary_s_matrix))

  println("Transposing teaching data")
  binary_s_matrix = transpose(binary_s_matrix)
  binary_a_matrix = transpose(binary_a_matrix)

  if cfg.use_progress_meter == true
    pm = Progress(rbm.numepochs, 1, "Training progress:", 50)
  end

  m     = ns - rbm.batchsize

  println("Starting learning")
  for t=1:rbm.numepochs
    # extract data batch for current epoch
    start = int(1 + floor(rand() * m)) # 1 to m
    if cfg.batchmode == "sequential"
      r     = [start:start+rbm.batchsize-1]
    else #if cfg.batch_mode == "random"
      r     = rand(1:ns, rbm.batchsize)
    end
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
      if t % cfg.plot_steps == 0 || t == 0
        clf()
        subplot(121)
        colorbar(imshow(rbm.W))
        subplot(122)
        colorbar(imshow(rbm.V))
      end
    end

    if cfg.use_progress_meter
      next!(pm)
    end
  end # training iteration
end

end
