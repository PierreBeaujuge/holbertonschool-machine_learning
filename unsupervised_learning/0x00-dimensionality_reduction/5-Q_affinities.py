#!/usr/bin/env python3
"""
5-Q_affinities.py
"""
import numpy as np


def Q_affinities(Y):
    """
    function that calculates the Q affinities
    """
    n, ndim = Y.shape

    # Compute the Q affinities
    # Start by computing the matrix D of squared pairwise distances
    # for the low dimensional dataset Y
    D = (np.sum(Y ** 2, axis=1) - 2 * np.matmul(Y, Y.T) +
         np.sum(Y ** 2, axis=1)[..., np.newaxis])
    D[[range(ndim)], range(ndim)] = 0
    A = 1 / (1 + D)
    # print("A.shape:", A.shape)
    A[range(n), range(n)] = 0
    B = np.sum(A)
    Q = A / B

    # Return the the numerator of the Q affinities as well
    num = A

    return Q, num
