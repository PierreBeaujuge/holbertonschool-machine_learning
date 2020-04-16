#!/usr/bin/env python3

cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

mat1 = [[1, 2], [3, 4]]
mat2 = [[5, 6]]
mat3 = [[7], [8]]
mat4 = cat_matrices2D(mat1, mat2)
mat5 = cat_matrices2D(mat1, mat3, axis=1)
print("mat1: {}".format(mat1))
print("mat2: {}".format(mat2))
print("mat3: {}".format(mat3))
print("mat4: {}".format(mat4))
print("mat5: {}".format(mat5))
mat1[0] = [9, 10]
print("upon executing mat1[0] = [9, 10]")
print("mat1: {}".format(mat1))
print("mat4: {}".format(mat4))
print("mat5: {}".format(mat5))
mat1[1].append(5)
print("upon executing mat1[1].append(5)")
print("mat1: {}".format(mat1))
print("mat4: {}".format(mat4))
print("mat5: {}".format(mat5))
