#!/usr/bin/env python3
"""define new function"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise"""
    # # making deep copies # here non-necessary
    # mat1 = [x[:] for x in mat1]
    # mat2 = [x[:] for x in mat2]
    # this if statement does not work; no explanation
    # if matrix_shape(mat1) == matrix_shape(mat2):
    if len(mat1) == len(mat2):
        mat3 = list(map(lambda arr1, arr2: add_arrays(arr1, arr2), mat1, mat2))
        if None in mat3:
            return None
        return mat3
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
