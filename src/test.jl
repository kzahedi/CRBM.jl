using CRBM
#using PyPlot

function run(S, A, bins, m, epochs, batch, alpha, uditer, momentum, weightcost, nr_of_sensors, perturbation, dropout)
  println("started job with $bins $m $epochs $batch $alpha $uditer $weightcost $momentum $nr_of_sensors $perturbation $dropout")
  n   = int(ceil(log2(bins))) * nr_of_sensors
  k   = int(ceil(log2(bins))) * nr_of_sensors

  rbm = rbm_create(n, m, k, uditer, alpha, momentum, weightcost, epochs, batch, bins, dropout)
  if perturbation > 0.0
    P  = perturbation .* repmat(randn(size(S)[1]), 1, size(S)[2])
    s  = S .* (1 .+ P)
    crbm_binary_train!(rbm, s, A, bins)
  else
    crbm_binary_train!(rbm, S, A, bins)
  end
  rbm_write("rbm.rbm", rbm)
  println("finished job rbm.rbm")
  rbm
end

data      = readdlm("data.log",' ')
sensors   = [1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12]
actuators = [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]
S         = data[:,sensors]
A         = data[:,actuators]

bins          = 16
m             = 20
epochs        = 1000
batch         = 10
alpha         = 0.1
uditer        = 10
momentum      = 0.5
weightcost    = 0.0
nr_of_sensors = 12
perturbation  = 0.0
dropout       = 0.0

# DAS IST DIE FUNKTION
rbm = run(S, A, bins, m, epochs, batch, alpha, uditer, momentum, weightcost, nr_of_sensors, perturbation, dropout)

#println(rbm.W)

#imshow(rbm.W)
#imshow(rbm.V)
#plot(rbm.b)
#plot(rbm.c)
