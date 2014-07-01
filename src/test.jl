using CRBM
#using PyPlot

function run(S, A, bins, m, epochs, batch, alpha, uditer, momentum, weightcost, nr_of_sensors, perturbation)
  println("started job $oindex/$all with $bins $m $epochs $batch $alpha $uditer $weightcost $momentum $nr_of_sensors $perturbation")
  n   = int(ceil(log2(bins))) * N
  k   = int(ceil(log2(bins))) * N

  rbm = rbm_create(n, m, k, uditer, alpha, momentum, weightcost, epochs, batch, bins)
  if perturbation > 0.0
    rbm_c = rbm_copy(rbm)
    P  = perturbation .* repmat(randn(size(S)[1]), 1, size(S)[2])
    s  = S .* (1 .+ P)
    crbm_binary_train_plain!(rbm, s, A, bins)
    println(sum(abs(rbm.W - rbm_c.W)))
  else
    rbm_c = rbm_copy(rbm)
    crbm_binary_train_plain!(rbm, S, A, bins)
    println(sum(abs(rbm.W - rbm_c.W)))
  end
  rbm_write("rbm-m_$m/rbm_$index.rbm", rbm)
  println("finished job rbm-m_$m/rbm_$index.rbm on $m")
  rbm
end

data      = readdlm("data.log")
sensors   = [1  2  3  4  5  6  7  8  9  10 11 12]
actuators = [58 59 60 61 62 63 64 65 66 67 68 69]
S         = data[:,sensors]
A         = data[:,actuators]

bins          =
m             =
epochs        =
batch         =
alpha         =
uditer        =
momentum      =
weightcost    =
nr_of_sensors =
perturbation  =

# DAS IST DIE FUNKTION
rbm = run(S, A, bins, m, epochs, batch, alpha, uditer, momentum, weightcost, nr_of_sensors, perturbation)

println(rbm.W)

#imshow(rbm.W)
#imshow(rbm.V)
#plot(rbm.b)
#plot(rbm.c)
