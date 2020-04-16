#!/usr/bin/env python3
"""define new function"""


def matrix_transpose(matrix):
    """function that returns the transpose of a matrix"""
    # return [[matrix[i][j] for j in range(len(matrix[i]))]
    # for i in range(len(matrix))]
    return [[matrix[j][i] for j in range(len(matrix))]
            for i in range(len(matrix[0]))]
