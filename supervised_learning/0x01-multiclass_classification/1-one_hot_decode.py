#!/usr/bin/env python3
"""
One-Hot Decode
"""
import numpy as np


def one_hot_decode(one_hot):
    """function that converts a one-hot matrix into a numeric label vector"""
    if not isinstance(one_hot, np.ndarray) or len(one_hot.shape) != 2:
        return None
    if not np.where((one_hot == 0) | (one_hot == 1), True, False).all():
        return None
    if np.sum(one_hot) != one_hot.shape[1]:
        return None
    return np.array([i for j in range(one_hot.shape[1])
                     for i in range(one_hot.shape[0]) if one_hot[i, j] == 1])

    # another option:
    # m = one_hot.shape[1]
    # labels = np.zeros(m)
    # index = np.arange(m)
    # label = np.argmax(one_hot, axis=0)
    # labels[index] = label
    # return labels.astype(int)
