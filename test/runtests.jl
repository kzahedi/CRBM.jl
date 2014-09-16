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
