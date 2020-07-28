#!/usr/bin/env python3
"""
0-mean_cov.py
"""
import numpy as np


def mean_cov(X):
    """function that calculates the mean and covariance matrix of a data set"""

    err_1 = "X must be a 2D numpy.ndarray"
    if not isinstance(X, np.ndarray):
        raise TypeError(err_1)
    if X.ndim != 2:
        raise TypeError(err_1)

    err_2 = "X must contain multiple data points"
    if X.shape[0] < 2:
        raise ValueError(err_2)

    n = X.shape[0]
    d = X.shape[1]

    mean = np.mean(X, axis=0)
    mean = mean[np.newaxis, ...]

    X = X - mean
    cov = np.matmul(X.T, X) / (n - 1)

    return mean, cov
