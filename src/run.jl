using CRBM
using PyPlot

discretise_value(v::Float64, nr_of_bins::Int64)             = int(min(floor(nr_of_bins * (v .+ 1) ./ 2.0), nr_of_bins-1))

function run_job!(rbm, S, A)
    N = size(sensors)[2]
    println("started job with $(rbm.bins) $(rbm.m) $(rbm.numepochs) $(rbm.batchsize) $(rbm.alpha) $(rbm.uditer) $(rbm.weightcost) $(rbm.momentum) $N $(rbm.perturbation)")

    n   = int(ceil(log2(bins))) * N
    k   = int(ceil(log2(bins))) * N
    
    if perturbation > 0.0
        P  = perturbation .* repmat(randn(size(S)[1]), 1, size(S)[2])
        s  = S .* (1 .+ P)
        crbm_binary_train!(rbm, s, A)
    else
        crbm_binary_train!(rbm, S, A)
    end
#    rbm_write("/Users/zahedi/Desktop/rbm.rbm", rbm)
    println("done.")
end

function undiscretise(v, nr_of_bins) # tested
  n = int(log2(nr_of_bins))
  l = int(length(v)/log2(nr_of_bins))
  r = zeros(l)
  for i=1:l
    r[i] = binary2integer(v[(i-1)*n+1:i*n])
  end
  r = 2.0 .* (r ./ (nr_of_bins) .+ 1 / (2*nr_of_bins)) .- 1.0
end

function binary2integer(v) # tested
  d = convert(Vector{Int64}, v)
  r = 0
  n = length(d)
  for i=1:n
    r += (d[i]>0)?(1<<(n-i)):0
  end
  return r
end

function plot_quality(myrbm)
    L   = 100
    out = zeros(L)
    ins = zeros(L)

    X = binarise_matrix(2.0 .* rand(1,12) .- 1.0, myrbm.bins)
    for i = 1:L
        s              = sensors[i+1,:]
        a              = actuators[i,:]    
        binned_sensors = binarise_matrix(s, myrbm.bins)
        nX             = crbm_control_sample!(myrbm, binned_sensors, X)
        o              = undiscretise(nX, myrbm.bins)
        X              = nX
        out[i]         = o[1]
        ins[i]         = s[1]
    end
    plot(ins)
    plot(out)
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

sensor_indices   = [1:12]
actuator_indices = [58:69];

data      = readdlm("data.log", ' ');
sensors   = data[:,sensor_indices];
actuators = data[:,actuator_indices];
data      = [];

bins          = 8;
units         = int(log2(bins)) # units pro sensor
n             = units * size(sensors)[2]
k             = n;
m             = 300;
uditer        = 15;
alpha         = 0.5;
momentum      = 0.5;
weightcost    = 0.00;
epochs        = 1000;
batch         = 50;
perturbation  = 0.00;

overallepochs = 0

rbm           = rbm_create(n, m, k, uditer, alpha, momentum, weightcost, epochs, batch, bins, 0.0, perturbation);


overallepochs = overallepochs + rbm.numepochs

run_job!(rbm, sensors, actuators)
println("After $overallepochs epochs")
println("Min/max V: $(minimum(rbm.V)) $(maximum(rbm.V))")
println("Min/max W: $(minimum(rbm.W)) $(maximum(rbm.W))")
println("Min/max b: $(minimum(rbm.b)) $(maximum(rbm.b))")
println("Min/max c: $(minimum(rbm.c)) $(maximum(rbm.c))")
plot_quality(rbm)
