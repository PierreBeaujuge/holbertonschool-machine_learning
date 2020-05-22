#!/usr/bin/env python3
"""
Shuffle Data
"""
import numpy as np


def shuffle_data(X, Y):
    """function that shuffles the data points in two matrices the same way"""
    shuffle = np.random.permutation(X.shape[0])
    # print(shuffle)
    X_shuf = X[shuffle]
    Y_shuf = Y[shuffle]
    return X_shuf, Y_shuf
