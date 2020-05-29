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
        # output layer: use softmax activation function
        if i == L - 1:
            cache['A' + str(i + 1)] = softmax(Zi)
        # hidden layers: use tanh activation function
        else:
            # all layers use tanh activation function, except last
            cache['A' + str(i + 1)] = tanh(Zi)
            # dropout mask applied to hidden layers only, "after" activation
            # define a dropout array of 1 and 0, with keep_prob 1s
            # dropout mask should be different for every layer
            drop = np.where((np.random.rand(cache['A' + str(i + 1)].shape[0],
                                            cache['A' + str(i + 1)].shape[1])
                             < keep_prob) == 1, 1, 0)
            # regularize by shutting off keep_prob outputs
            # and normalize by keep_prob (important)
            cache['A' + str(i + 1)] *= drop
            cache['A' + str(i + 1)] /= keep_prob
            cache['D' + str(i + 1)] = drop
    return cache


def tanh(Y):
    """define the tanh activation function"""
    return np.tanh(Y)


def softmax(Y):
    """define the softmax activation function"""
    return np.exp(Y) / (np.sum(np.exp(Y), axis=0, keepdims=True))
