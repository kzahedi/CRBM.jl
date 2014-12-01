module CRBM

include("conversion_functions.jl")
include("update_functions.jl")

using RBM
using ProgressMeter
#using PyPlot

export crbm_control_sample
export crbm_binary_train!

#export from RBM package, so that you don't have to import both
export RBM_t
export CRBM_cfg_t, crbm_create_config
export rbm_copy
export rbm_create
export rbm_init_weights_random!
export rbm_init_visible_bias!
export rbm_init_output_bias_random!, rbm_init_hidden_bias_random!
export rbm_write, rbm_read, rbm_read_old

export binarise_matrix, i2b, iv2b, binarise_vector
export unbinarise_matrix, b2i, b2iv, bv2dv
export bv, bin_matrix, bin_vector
export unbin_value, unbin_matrix, unbin_vector
export binary_draw
export up, down, binary_up, binary_down

type CRBM_cfg_t
  use_progress_meter::Bool
  use_pyplot::Bool
  plot_steps::Int64
  batchmode::String
  verbose::Bool
end

function crbm_create_config()
  return CRBM_cfg_t(true, true, 100, "random", false)
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
  return X,Z
end

function crbm_binary_train!(cfg::CRBM_cfg_t, rbm::RBM_t, S::Matrix{Float64}, A::Matrix{Float64})
  @assert (rbm.dropout >= 0.0 && rbm.dropout <= 1.0) "Dropout must be in [0,1]"

  # TODO more assert statements needed
  N  = ceil(log2(rbm.bins))
  if cfg.verbose == true
    println("binarising sensor data")
  end
  binary_s_matrix = binarise_matrix(S, rbm.bins)
  if cfg.verbose == true
    println("binarising actuator data")
  end
  binary_a_matrix = binarise_matrix(A, rbm.bins)
  ns = size(S)[1]
  nb = rbm.batchsize

  if maximum(rbm.W) == 0.0 && minimum(rbm.W) == 0.0 &&
     maximum(rbm.V) == 0.0 && minimum(rbm.V) == 0.0
     if cfg.verbose == true
       println("Initialising W, V, and c.")
     end
    rbm_init_weights_random!(rbm)
    rbm.c = zeros(rbm.m)
  end

  if cfg.verbose == true
    println("Initialising visible bias")
  end
  rbm_init_visible_bias!(rbm, convert(Array{Int64},binary_s_matrix))

  if cfg.use_progress_meter == true
    pm = Progress(rbm.numepochs, 1, "Training progress:", 20)
  end

  m = ns - rbm.batchsize

  if cfg.verbose == true
    println("Starting learning")
  end
  for t=1:rbm.numepochs
    # extract data batch for current epoch
    start = int(1 + floor(rand() * m)) # 1 to m
    if cfg.batchmode == "sequential"
      r     = [start:start+rbm.batchsize-1]
    else #if cfg.batch_mode == "random"
      r     = rand(1:ns, rbm.batchsize)
    end

    s = binary_s_matrix[r,:]
    a = binary_a_matrix[r,:]

    # generate hidden states given the data
    #= z = binary_up(rbm, s, a) =#
    z = binary_up(rbm, s, a)

    # generate random outputs to start sampler
    A=zeros(size(s)[1], rbm.n)
    for i=1:size(s)[1]
      A[i,:] = transpose(i2b(int(floor(2^rbm.n * rand())), rbm.n ))
    end

    (A, Z) = crbm_learn_sampling(rbm, s, A)

    Eb  = transpose(mean(a,1) - mean(A,1))
    Ec  = transpose(mean(z,1) - mean(Z,1))
    EW  = (z' * a - Z' * A)/nb
    EV  = (z' * s - Z' * s)/nb

    Eb = squeeze(Eb,2)
    Ec = squeeze(Ec,2)

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

    #= if cfg.use_pyplot == true =#
      #= if t % cfg.plot_steps == 0 || t == 0 =#
        #= clf() =#
        #= subplot(121) =#
        #= colorbar(imshow(rbm.W)) =#
        #= subplot(122) =#
        #= colorbar(imshow(rbm.V)) =#
      #= end =#
    #= end =#

    if cfg.use_progress_meter
      next!(pm)
    end
  end # training iteration
end

end
