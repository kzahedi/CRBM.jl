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
