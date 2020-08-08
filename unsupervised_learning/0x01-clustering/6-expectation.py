#!/usr/bin/env python3
"""
6-expectation.py
"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    function that calculates the expectation step in the EM algorithm for a GMM
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or pi.ndim != 1:
        return None, None
    if not isinstance(m, np.ndarray) or m.ndim != 2:
        return None, None
    if not isinstance(S, np.ndarray) or S.ndim != 3:
        return None, None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    if pi.shape[0] > n:
        return None, None
    k = pi.shape[0]
    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    # Ensure the sum of all priors is equal to 1
    if not np.isclose([np.sum(pi)], [1])[0]:
        return None, None

    # Initialize the array of posteriors "g"
    g = np.zeros((k, n))
    # Iterate over each cluster centroid:
    for i in range(k):
        # Compute the PDF
        # (here a 1D array containing the PDF values for each data point)
        PDF = pdf(X, m[i], S[i])
        # Compute the posteriors
        # (here a 1D array of n posterior values)
        # (pi[i] is a single prior value picked from pi:
        # 1D array containing the priors for each cluster)
        g[i] = pi[i] * PDF
        # print("g[{}]:".format(i), g[i])
    # Sum across the k clusters; set keepdims=True to adress checker reqs
    sum_gis = np.sum(g, axis=0, keepdims=True)
    # print("sum_gis:", sum_gis)
    # print("sum_gis.shape:", sum_gis.shape)
    g /= sum_gis
    # print("g.shape:", g.shape)

    # Compute the likelihood
    lkhd = np.sum(np.log(sum_gis))

    return g, lkhd
