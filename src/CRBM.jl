module CRBM

using RBM

export crbm_control_sample!
export crbm_binary_train!

#export from RBM package, so that you don't have to import both
export RBM_t
export rbm_copy
export rbm_create
export rbm_write, rbm_read

sigm(p::Matrix{Float64})                                    = 1./(1 .+ exp(-p))
binary_draw(p::Matrix{Float64})                             = p .> rand(size(p))
binary_up(rbm::RBM_t, y::Array{Float64}, x::Array{Float64}) = convert(Matrix{Float64},binary_draw(up(rbm, y, x)))
binary_down(rbm::RBM_t, z::Array{Float64})                  = convert(Matrix{Float64},binary_draw(down(rbm, z)))
discretise_value(v::Float64, nr_of_bins::Int64)        = int(min(floor(nr_of_bins * (v .+ 1) ./ 2.0), nr_of_bins-1))

function down(rbm::RBM_t, z::Array{Float64})
  r = sigm(repmat(rbm.b, 1, size(z)[1]) + rbm.W' * z')
  r'
end

function up(rbm::RBM_t, y::Array{Float64}, x::Array{Float64})
  r = sigm(repmat(rbm.c, 1, size(y)[1]) + rbm.V * y' + rbm.W * x')
  r'
end

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
      d     = discretise_value(value, bins)
      b     = int2binary(d, N)
      for u = 1:N
        B[i,(j-1)*N+u] = float64(b[u])
      end
    end
  end
  B
end

function crbm_binary_train!(rbm, S, A)
  @assert (rbm.dropout >= 0.0 && rbm.dropout <= 1.0) "Dropout must be in [0,1]"
  # TODO more asserts
  N  = ceil(log2(rbm.bins))
  ss = binarise_matrix(S, rbm.bins)
  aa = binarise_matrix(A, rbm.bins)

  if maximum(rbm.W) == 0.0 && minimum(rbm.W) == 0 && maximum(rbm.W) == 0.0 && minimum(rbm.W) == 0
    println("Initialising W,V, and c.")
    rbm_init_weights_random!(rbm)
    rbm.c           = zeros(rbm.m)
  end

  binary_s_matrix = binarise_matrix(S, rbm.bins)
  rbm_init_visible_bias!(rbm, convert(Array{Int64},binary_s_matrix))

  for t=1:rbm.numepochs
    # extract data batch for current epoch
    m     = size(ss)[1] - rbm.batchsize
    start = int(1 + floor(rand() * m)) # 1 to m
    r     = [start:start+rbm.batchsize]
    s     = ss[r,:]
    a     = aa[r,:]

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

    rbm.b = rbm.b + rbm.alpha * Eb
    rbm.c = rbm.c + rbm.alpha * Ec
    rbm.W = rbm.W + rbm.alpha * EW
    rbm.V = rbm.V + rbm.alpha * EV

    if rbm.momentum > 0.0
      rbm.b = rbm.b + (rbm.alpha * rbm.momentum) * rbm.vb
      rbm.c = rbm.c + (rbm.alpha * rbm.momentum) * rbm.vc
      rbm.W = rbm.W + (rbm.alpha * rbm.momentum) * rbm.vW
      rbm.V = rbm.V + (rbm.alpha * rbm.momentum) * rbm.vV
    end

    if rbm.weightcost > 0.0 # using L2
      rbm.W = rbm.W * (1 - rbm.weightcost)
      rbm.V = rbm.V * (1 - rbm.weightcost)
    end

    rbm.vb = Eb
    rbm.vc = Ec
    rbm.vW = EW
    rbm.vV = EV

  end # training iteration
end

end


