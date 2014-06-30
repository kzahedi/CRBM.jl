module CRBM

using RBM

export RBM_t, rbm_copy
export rbm_create, rbm_create_with_standard_values
export rbm_init_weights_random!, rbm_init_visible_bias!
export rbm_init_output_bias_random!, rbm_init_hidden_bias_random!
export rbm_rescale_weights!
export rbm_calculate_L1, rbm_calculate_L2
export rbm_write, rbm_read
export sigm
export rbm_visualise

export crbm_control_sample!
export crbm_binary_train_plain!

# TODO: Implement this functions
# export crbm_binary_train_L1!
# export crbm_binary_train_L2!
# export crbm_binary_train_weight_scaling!
# END TODO

sigm(p::Matrix{Float64})                                    = 1./(1 .+ exp(-p))
up(rbm::RBM_t, y::Array{Float64}, x::Array{Float64})        = sigm(repmat(rbm.c, 1, size(y)[1]) + rbm.V * y' + rbm.W * x')
down(rbm::RBM_t, z::Array{Float64})                         = sigm(repmat(rbm.b, 1, size(z)[1]) + rbm.W' * z')
binary_draw(p::Matrix{Float64})                             = p .> rand(size(p))
binary_up(rbm::RBM_t, y::Array{Float64}, x::Array{Float64}) = convert(Matrix{Float64},up(rbm, y, x)')
binary_down(rbm::RBM_t, z::Array{Float64})                  = convert(Matrix{Float64},binary_draw(down(rbm, z)'))

function crbm_learn_sampling!(rbm::RBM_t, y::Array{Float64}, X::Array{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end       
  X = down(rbm, Z)
  Z = up(rbm, y, X)
  return X,Z
end

function crbm_control_sample!(rbm::RBM_t, y::Array{Float64}, X::Array{Float64})
  Z = binary_up(rbm, y, X)
  for i=1:rbm.uditer-1
    X = down(rbm, Z)
    Z = binary_up(rbm, y, X)
  end       
  X = binary_down(rbm, Z)
  return X
end

function int2binary(v::Int64, n::Int64) # checked
  r=zeros(n)
  for i=1:n
    r[i] = (((1 << (n-i)) & v)>0)?1.0:0.0
  end
  return r
end

function binarise_matrix(A::Matrix{Float64}, bins::Int64)
  N = int(ceil(log2(bins)))
  B=zeros(size(A)[1], size(A)[2]* N)
  for i=1:size(A)[1]
    for j=1:size(A)[2]
      value = A[i,j]
      d     = dvalue(value, bins)
      b     = int2binary(d, N)
      for u = 1:N
        B[i,(j-1)*N+u] = b[u]
      end
    end
  end
  B
end

function crbm_binary_train_plain!(rbm, S, A, bins, perturbation)
  N  = ceil(log2(bins))
  P  = perturbation .* randn(size(S))
  ss = binarise_matrix(S + P, bins)
  aa = binarise_matrix(A, bins)

  rbm_init_weights_random!(rbm)
  rbm.c           = zeros(rbm.m)
  binary_s_matrix = binarise_matrix(S, bins)
  rbm_init_visible_bias!(rbm, binary_s_matrix)

  for t=1:rbm.numepochs
    # extract data batch for current epoch
    m     = length(ss) - rbm.batchsize
    start = 1 + floor((rand() * m)) # 1 to m
    r     = [1:rbm.batchsize] .+ start
    s     = ss[int64(ceil(size(ss)[1] * r)),:]
    a     = aa[int64(ceil(size(aa)[1] * r)),:]

    # generate hidden states given the data
    z = up(rbm, s, a) 

    # generate random outputs to start sampler
    (A, Z) = crbm_learn_sampling!(rbm, s, a) 

    Eb  = transpose(mean(a,1) - mean(A,1))
    Ec  = transpose(mean(z,1) - mean(Z,1))
    EW  = (z' * a - Z' * A)/size(s)[1]
    EV  = (z' * s - Z' * s)/size(s)[1]

    Eb = squeeze(Eb,2)
    Ec = squeeze(Ec,2)

    if rbm.momentum == 0
      rbm.b = rbm.b + rbm.alpha * Eb  
      rbm.c = rbm.c + rbm.alpha * Ec  
      rbm.W = rbm.W + rbm.alpha * EW  
      rbm.V = rbm.V + rbm.alpha * EV     
    else 
      rbm.b = rbm.b + rbm.alpha * Eb + rbm.momentum * rbm.vb 
      rbm.c = rbm.c + rbm.alpha * Ec + rbm.momentum * rbm.vc 
      rbm.W = rbm.W + rbm.alpha * EW + rbm.momentum * rbm.vW 
      rbm.V = rbm.V + rbm.alpha * EV + rbm.momentum * rbm.vV

      rbm.vb = Eb
      rbm.vc = Ec
      rbm.vW = EW
      rbm.vV = EV
    end

  end # training iteration
end 

end


