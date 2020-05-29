#!/usr/bin/env python3
"""
Forward Propagation with Dropout
"""
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """function that conducts forward propagation using Dropout"""
    cache = {}
    cache['A0'] = X
    for i in range(L):
        Zi = np.matmul(
            weights['W' + str(i + 1)], cache['A' + str(i)]
        ) + weights['b' + str(i + 1)]
        if i == L - 1:
            cache['A' + str(i + 1)] = np.exp(Zi) / (
                np.sum(np.exp(Zi), axis=0, keepdims=True))
        else:
            cache['A' + str(i + 1)] = np.tanh(Zi)
            boolean = np.random.rand(
                cache['A' + str(i + 1)].shape[0],
                cache['A' + str(i + 1)].shape[1]) < keep_prob
            drop = np.where(boolean == 1, 1, 0)
            cache['A' + str(i + 1)] *= drop
            cache['A' + str(i + 1)] /= keep_prob
            cache['D' + str(i + 1)] = drop
    return cache
