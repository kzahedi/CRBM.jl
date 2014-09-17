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

binarise_vector(v::Vector{Float64}, bins::Int64) = iv2b(bin_vector(v, -1.0, 1.0, bins), int(ceil(log2(bins))))

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

# 
# end   - covered by test cases
#
