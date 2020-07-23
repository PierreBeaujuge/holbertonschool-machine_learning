#!/usr/bin/env python3
"""Advanced Linear Algebra"""
import numpy as np


def definiteness(matrix):
    """function that determines the definiteness of a matrix"""

    err_1 = "matrix must be a numpy.ndarray"
    if not isinstance(matrix, np.ndarray):
        raise TypeError(err_1)
    if matrix.ndim != 2:
        return None
    if matrix.shape[0] == 0 or matrix.shape[1] == 0:
        return None
    if matrix.shape[0] != matrix.shape[1]:
        return None

    # Matrix must be symmetric
    # if not the case, return None
    if not np.array_equal(matrix, matrix.T):
        return None

    # Infer the number of rows/columns in matrix
    height = matrix.shape[0]
    width = matrix.shape[1]

    # Define an empty list in which all the calculated
    # determinants will be added
    D = []

    # Iterate through the row and column indices (concurrently)
    for row_index, col_index in zip(range(height), range(width)):
        # Call the numpy det() method on each submat
        det = np.linalg.det(matrix[:row_index + 1, :col_index + 1])
        # print(type(det), det)
        # Add the determinant to D
        D.append(det)

    # Convert the list to a np.ndarray
    D = np.array(D)
    # print(D)

    # Positive definite
    if all(D > 0):
        return "Positive definite"

    # Negative definite
    if all(D[0::2] < 0) and all(D[1::2] > 0):
        return "Negative definite"

    # Indefinite: if neither Positive, nor Negative definite
    # and if det(matrix) != 0
    if D[-1] != 0:
        return "Indefinite"

    # Positive semi-definite: if det(matrix) == 0
    # and all(D > 0)
    if D[-1] == 0 and all(D[:-1] > 0):
        return "Positive semi-definite"

    # Negative semi-definite: if det(matrix) == 0
    # and all(D > 0)
    if D[-1] == 0 and all(D[0:-1:2] < 0) and all(D[1:-1:2] > 0):
        return "Negative semi-definite"
