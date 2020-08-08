#!/usr/bin/env python3
"""
7-maximization.py
"""
import numpy as np


def maximization(X, g):
    """
    function that calculates the maximization step
    in the EM algorithm for a GMM
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None

    # X: array of shape (n, d) containing the data set
    n, d = X.shape

    # g: array of shape (k, n) containing the posteriors
    if g.shape[1] != n:
        return None, None, None
    k = g.shape[0]
    if g.shape[0] != k:
        return None, None, None

    # Ensure the sum of all posteriors (over the k clusters) is equal to 1
    if not np.isclose(np.sum(g, axis=0), np.ones(n,)).all():
        return None, None, None

    # Initialize pi, m and S with zeros
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    # Iterate over each cluster centroid:
    for i in range(k):
        # Sum gi over the n data points
        gn = np.sum(g[i], axis=0)
        pi[i] = gn / n
        m[i] = np.sum(np.matmul(g[i][np.newaxis, ...], X), axis=0) / gn
        S[i] = np.matmul(g[i][np.newaxis, ...] * (X - m[i]).T, (X - m[i])) / gn
        # print("pi[{}]:".format(i), pi[i])
        # print("m[{}]:".format(i), m[i])
        # print("S[{}]:".format(i), S[i])

    return pi, m, S
