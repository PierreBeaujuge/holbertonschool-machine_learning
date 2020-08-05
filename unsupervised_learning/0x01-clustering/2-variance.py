#!/usr/bin/env python3
"""
2-variance.py
"""
import numpy as np


def variance(X, C):
    """
    function that calculates the total intra-cluster variance for a data set
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    # Number of cluster centroids of shape (k, d)
    k = C.shape[0]

    if k > n:
        return None
    if C.shape[1] != d:
        return None

    # OPTION 1: FOR LOOPS

    # Initialize the array of pairwise data point-centroid
    # distances with zeros
    # dist = np.zeros((n, k))

    # Compute the "dist" matrix of euclidean distances between
    # data points and centroids
    # for i in range(n):
    #     for j in range(k):
    #         dist[i, j] = np.linalg.norm(X[i, ...] - C[j, ...])

    # OPTION 2: VECTORIZATION

    # Convert X into an array suitable for vectorization
    Xv = np.repeat(X, k, axis=0)
    Xv = Xv.reshape(n, k, d)

    # Convert C into an array suitable for vectorization
    Cv = np.tile(C, (n, 1))
    Cv = Cv.reshape(n, k, d)

    # Compute the "dist" matrix of euclidean distances between
    # data points and centroids; shape (n, k)
    dist = np.linalg.norm(Xv - Cv, axis=2)

    # Determine the centroid to which each data point relates to
    # clss = np.argmin(dist, axis=1)
    # Compute the 1D array of shortest data point-centroid
    # squared distances
    short_dist = np.min(dist ** 2, axis=1)
    # print("short_dist:", short_dist)
    # print("short_dist.shape:", short_dist.shape)

    # Sum of "short_dist" over the n data points == definition of
    # overall intra-cluster variance for the dataset
    # Evaluate the variance of the corresponding array
    var = np.sum(short_dist)

    return var
