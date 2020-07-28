#!/usr/bin/env python3
"""
1-correlation.py
"""
import numpy as np


def correlation(C):
    """function that calculates the correlation matrix of a data set"""

    err_1 = "C must be a numpy.ndarray"
    if not isinstance(C, np.ndarray):
        raise TypeError(err_1)

    err_2 = "C must be a 2D square matrix"
    if C.ndim != 2:
        raise ValueError(err_2)
    if C.shape[0] != C.shape[1]:
        raise ValueError(err_2)

    D = np.sqrt(np.diag(np.diag(C)))
    D_inv = np.linalg.inv(D)

    corr = np.linalg.multi_dot([D_inv, C, D_inv.T])

    return corr
