#!/usr/bin/env python3
"""
One-Hot Encode
"""
import numpy as np


def one_hot_encode(Y, classes):
    """function that converts a numeric label vector into a one-hot matrix"""
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None
    if not isinstance(classes, int) or classes < np.max(Y) + 1:
        return None
    return np.array([[1 if j == x else 0 for j in range(classes)]
                     for x in Y.tolist()]).astype(float).T

    # another option:
    # m = Y.shape[0]
    # one_hot = np.zeros((classes, m))
    # y_indices = np.arange(m)
    # x_indices = Y
    # one_hot[x_indices, y_indices] = 1
    # return one_hot
