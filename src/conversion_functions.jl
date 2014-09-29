# 
# start - covered by test cases
#

bv(v::Float64, bins::Int64, min=-1.0, max=1.0) = minimum([int64(floor(bins * (v - min) / (max - min))), bins-1])

bin_vector(vec::Vector{Float64}, min::Float64, max::Float64, bins::Int64) = map(v->bv(v, bins, min, max), vec)

function unbin_value(v::Int64, bins::Int64, min=-1.0, max=1.0; mode="centre")
  known_mode = (mode == "centre" ||
                mode == "lower"  ||
                mode == "upper")
  @assert known_mode "Mode may be any of the following: [\"centre\", \"lower\", \"upper\"]"

  delta = (max - min) / float64(bins)
  u = (v - 1) * delta + min

  if     mode == "centre"
    return u + 0.5 * delta
  elseif mode == "upper"
    return u + delta
  end
  return u
end

function bin_matrix(m::Matrix{Float64}, min::Float64, max::Float64, bins::Int64)
  r = zeros(size(m))
  for i=1:size(m)[1]
    r[i,:] = bin_vector(squeeze(m[i,:],1), min, max, bins)
  end
  convert(Matrix{Int64}, r)
end

function unbin_matrix(m::Matrix{Float64}, min::Float64, max::Float64, bins::Int64; mode="centre")
  r = zeros(size(m))
  for i=1:size(m)[1]
    r[i,:] = unbin_vector(squeeze(m[i,:],1), min, max, bins, mode=mode)
  end
  convert(Matrix{Int64}, r)
end



unbin_vector(vec::Vector{Float64}, min::Float64, max::Float64, bins::Int64; mode="centre") = map(v->unbin_value(v, bins, min, max, mode=mode), vec)


i2b(v::Int64, n::Int64) = [(((1 << (n-i)) & v)>0)?1.0:0.0 for i=1:n]
b2i(v::Vector{Float64}) = int64(foldl(+, [(v[i]>0)?(1<<(length(v)-i)):0 for i=1:length(v)]))

iv2b(v::Vector{Int64},   n::Int64) = foldl(vcat, [i2b(u ,n) for u in v])
b2iv(v::Vector{Float64}, n::Int64) = foldl(vcat, [b2i(v[i:i+(n-1)]) for i=1:n:length(v)])

bv2dv(v::Vector{Float64}, bins::Int64; mode="centre") = map(x->unbin_value(int64(x+1), bins, -1.0, 1.0, mode=mode), b2iv(v, int(ceil(log2(bins)))))

function binarise_matrix(A::Matrix{Float64}, bins::Int64)
  N = int(ceil(log2(bins)))
  B = zeros(size(A)[1], size(A)[2]* N)
  C = bin_matrix(A, -1.0, 1.0, bins)
  for row_index = 1:size(A)[1]
    B[row_index,:] = iv2b(squeeze(C[row_index,:],1), N)
  end
  B
end

# .- because bin_vector in [1, bins]
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
