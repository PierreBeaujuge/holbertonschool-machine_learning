#!/usr/bin/env python3
"""define new function"""


def add_arrays(arr1, arr2):
    """function that adds two arrays element-wise"""
    # if matrix_shape(arr1) == matrix_shape(arr2):
    if len(arr1) == len(arr2):
        # return list(map(lambda x, y: x + y, arr1, arr2))
        return [arr1[i] + arr2[i] for i in range(len(arr1))]
    return None


def matrix_shape(matrix):
    """function that returns the shape of a matrix"""
    shape = [len(matrix)]
    while isinstance(matrix[0], list):
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape
