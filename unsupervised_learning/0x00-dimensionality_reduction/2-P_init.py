#!/usr/bin/env python3
"""
2-P_init.py
"""
import numpy as np


def P_init(X, perplexity):
    """
    function that initializes variables used to calculate
    the P affinities in t-SNE
    """
    n, d = X.shape

    # Compute the matrix D of squared pairwise distances
    # between two data points; use the following resource:
    # https://medium.com/@souravdey/
    # l2-distance-matrix-vectorization-trick-26aa3247ac6c
    # exploit: (a - b) ** 2 = a ** 2 - 2ab + b ** 2
    D = (np.sum(X ** 2, axis=1) - 2 * np.matmul(X, X.T) +
         np.sum(X ** 2, axis=1)[..., np.newaxis])
    # Taking the square of the Frobenius norm also works:
    # D = (np.linalg.norm(X, axis=1) ** 2 - 2 * np.matmul(X, X.T) +
    #      (np.linalg.norm(X, axis=1) ** 2)[..., np.newaxis])

    # Note: the distance from one point to itself should be 0;
    # here these distances live in the diagonal -> ensure 0s
    # Set zeros across the matrix diagonal:
    D[[range(n)], range(n)] = 0

    # Initialize the array of P affinities:
    P = np.zeros((n, n))

    # Initialize the array of betas:
    betas = np.ones((n, 1))

    # Shannon entropy for perplexity:
    H = np.log2(perplexity)

    return D, P, betas, H
