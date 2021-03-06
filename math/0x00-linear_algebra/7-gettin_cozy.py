#!/usr/bin/env python3
"""define new function"""


def cat_matrices2D(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis"""
    # making deep copies
    mat1 = [x[:] for x in mat1]
    mat2 = [x[:] for x in mat2]
    # edge case not tested here:
    # if axis not in range(len(matrix_shape(mat1))):
    #     return None
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        return cat_arrays(mat1, mat2)
    elif axis == 1 and len(mat1) == len(mat2):
        return list(map(lambda arr1, arr2: cat_arrays(arr1, arr2), mat1, mat2))
    else:
        return None


def cat_arrays(arr1, arr2):
    """function that adds two matrices element-wise"""
    return arr1 + arr2


def matrix_shape(matrix):
    """function that returns the shape of a matrix"""
    shape = [len(matrix)]
    while isinstance(matrix[0], list):
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape
