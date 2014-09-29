using CRBM
using Base.Test

# write your own tests here
@test 1 == 1

# test basic functions - integer to binary vector
@test [0.0, 0.0, 0.0, 0.0] == i2b(0 ,4)
@test [0.0, 0.0, 0.0, 1.0] == i2b(1 ,4)
@test [0.0, 0.0, 1.0, 0.0] == i2b(2 ,4)
@test [0.0, 0.0, 1.0, 1.0] == i2b(3 ,4)
@test [0.0, 1.0, 0.0, 0.0] == i2b(4 ,4)
@test [0.0, 1.0, 0.0, 1.0] == i2b(5 ,4)
@test [0.0, 1.0, 1.0, 0.0] == i2b(6 ,4)
@test [0.0, 1.0, 1.0, 1.0] == i2b(7 ,4)
@test [1.0, 0.0, 0.0, 0.0] == i2b(8 ,4)
@test [1.0, 0.0, 0.0, 1.0] == i2b(9 ,4)
@test [1.0, 0.0, 1.0, 0.0] == i2b(10,4)
@test [1.0, 0.0, 1.0, 1.0] == i2b(11,4)
@test [1.0, 1.0, 0.0, 0.0] == i2b(12,4)
@test [1.0, 1.0, 0.0, 1.0] == i2b(13,4)
@test [1.0, 1.0, 1.0, 0.0] == i2b(14,4)
@test [1.0, 1.0, 1.0, 1.0] == i2b(15,4)

# test basic functions - integer vector to binary vector
@test [0.0, 0.0, 0.0, 0.0] == iv2b([0, 0], 2)
@test [0.0, 1.0, 1.0, 0.0] == iv2b([1, 2], 2)
@test [0.0, 1.0, 0.0, 0.0] == iv2b([4],    4)

# test basic functions - float matrix to binary vector
@test [[0.0 0.0 0.0 0.0], [0.0 0.0 0.0 0.0]] == binarise_matrix([[-1.0 -1.0], [-1.0 -1.0]], 4)

# test inverse binning
@test 0 == b2i([0.0, 0.0])
@test 1 == b2i([0.0, 1.0])
@test 2 == b2i([1.0, 0.0])
@test 3 == b2i([1.0, 1.0])

# test inverse binning - vector
@test [0, 1] == b2iv([0.0, 0.0, 0.0, 1.0], 2)
@test [2, 3] == b2iv([1.0, 0.0, 1.0, 1.0], 2)

@test [-0.75, -0.25] == bv2dv([0.0, 0.0, 0.0, 1.0], 4)
@test [0.25, 0.75]   == bv2dv([1.0, 0.0, 1.0, 1.0], 4)

# test inverse binning - matrix
@test [[-0.75 -0.25], [0.25 0.75]] == unbinarise_matrix([[0.0 0.0 0.0 1.0], [1.0 0.0 1.0 1.0]], 4)
 
# test binary_draw
@test [1.0 1.0 1.0 1.0 0.0 0.0 0.0 0.0] == binary_draw([1.1 1.1 1.1 1.1 -0.1 -0.1 -0.1 -0.1])


# test against old code

function undiscretise(v, nr_of_bins)
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

for i=0:200
  j = float(i-100)/200.0
  k = bv(j, 16, -1.0, 1.0)
  l = i2b(k,4)
  @test undiscretise(l, 16)[1] == bv2dv(l, 16)
end

function int2binary(v::Int64, n::Int64) # checked
  r=zeros(n)
  for i=1:n
    r[i] = (((1 << (n-i)) & v)>0)?1.0:0.0
  end
  return r
end

function discretise(v, nr_of_bins) # checked
  d = convert(Vector{Int64},min(floor(nr_of_bins * (v .+ 1) ./ 2.0), nr_of_bins-1))
  r = Float64[]
  u = int(log2(nr_of_bins))
  for i = 1:length(d)
    r = append!(r,int2binary(d[i], u))
  end
  return r
end

for i=0:200
  j = float(i-100)/200.0
  k = discretise([j], 16)
  l = iv2b(bin_vector([j], -1.0, 1.0, 16), 4)
  @test k == l
end


for i=1:100
  r = rand(12)
  #= @test discretise(r, 16) == binarise_vector(r, 16) =#
  @test discretise(r, 16) == iv2b(bin_vector(r, -1.0, 1.0, 16), 4)
end
