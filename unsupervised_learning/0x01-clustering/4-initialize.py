#!/usr/bin/env python3
"""
4-initialize.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """function that initializes variables for a Gaussian Mixture Model"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape
    # print(X.shape)
    # print(X)

    if not isinstance(k, int) or k <= 0 or k > n:
        return None, None, None

    # Initialize the "pi" array of shape (k,)
    # containing the priors for each cluster
    pi = np.full(shape=(k,), fill_value=1/k)

    # Initialize the "m" array of shape (k, d) containing
    # the centroid means for each cluster, initialized with K-means;
    # output is an array of coordinates
    m, _ = kmeans(X, k)

    # Initialize the "S" array of shape (k, d, d) containing
    # the covariance matrices for each cluster,
    # initialized as identity matrices
    Sij = np.diag(np.ones(d))
    S = np.tile(Sij, (k, 1)).reshape(k, d, d)

    return pi, m, S
