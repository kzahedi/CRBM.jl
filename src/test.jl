using CRBM

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
end
