#!/usr/bin/env python3


def add_matrices(mat1, mat2):
    """function that adds two matrices"""
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    if len(matrix_shape(mat1)) == 1:
        return [mat1[i] + mat2[i] for i in range(len(mat1))]
    elif len(matrix_shape(mat1)) == 2:
        return [[mat1[i][j] + mat2[i][j] for j in range(len(mat1[0]))]
                for i in range(len(mat1))]
    # elif len(matrix_shape(mat1)) == 3:
    #     return [[[mat1[i][j][k] + mat2[i][j][k]
    # for k in range(len(mat1[0][0]))]
    # for j in range(len(mat1[0]))] for i in range(len(mat1))]
    # elif len(matrix_shape(mat1)) == 4:
    #     return [[[[mat1[i][j][k][l] + mat2[i][j][k][l]
    # for l in range(len(mat1[0][0][0]))]
    # for k in range(len(mat1[0][0]))] for j in range(len(mat1[0]))]
    # for i in range(len(mat1))]
    elif len(matrix_shape(mat1)) >= 3:
        return [add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))]


def matrix_shape(matrix):
    """function that returns the shape of a matrix"""
    shape = [len(matrix)]
    while isinstance(matrix[0], list):
        shape.append(len(matrix[0]))
        matrix = matrix[0]
    return shape
