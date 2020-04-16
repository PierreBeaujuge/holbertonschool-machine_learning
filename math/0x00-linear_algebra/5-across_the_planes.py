#!/usr/bin/env python3
"""define new function"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""
    # making deep copies
    mat1 = [x[:] for x in mat1]
    mat2 = [x[:] for x in mat2]
    if matrix_shape(mat1) == matrix_shape(mat2):
        return list(map(lambda arr1, arr2: add_arrays(arr1, arr2), mat1, mat2))
    return None


def add_arrays(arr1, arr2):
    """function that adds two arrays element-wise"""
    if len(arr1) == len(arr2):
        return list(map(lambda x, y: x + y, arr1, arr2))
    return None


def matrix_shape(matrix):
    """function that returns the shape of a matrix"""
    shape = [len(matrix)]
    while isinstance(matrix[0], list):
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape
