#!/usr/bin/env python3
"""
3-optimum.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """function that tests for the optimum number of clusters by variance"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0 or n <= kmin:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or n < kmax:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize list of tuples (C, clss)
    results = []
    # Initialize list of total intra-cluster variances
    variances = []
    # Initialize the list of difference in variance from
    # the smallest cluster size for each cluster size
    d_vars = []

    # Iterate over the number of clusters under consideration
    for k in range(kmin, kmax + 1):

        # Compute the cluster centroids C (means; coordinates and the
        # 1D array of data point-centroid assignement in a call to kmeans()
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        # Compute the corresponding total intra-cluster variance
        var = variance(X, C)
        variances.append(var)

    for var in variances:
        d_vars.append(np.abs(variances[0] - var))

    return results, d_vars
