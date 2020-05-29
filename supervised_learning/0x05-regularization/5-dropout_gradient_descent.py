#!/usr/bin/env python3
"""
Gradient Descent with Dropout
"""
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """function that updates the weights and biases of a nn using
    gradient descent with Dropout"""
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        m = Y.shape[1]
        if i != L:
            # all layers use a tanh activation, except last
            # introduce call to tanh_prime method
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), tanh_prime(cache['A' + str(i)]))
            # pass dZi through same dropout mask as that
            # saved in cache during forward_prop
            # dropout mask applied to hidden layers only
            # regularize and normalize by keep_prob
            dZi *= cache['D' + str(i)]
            dZi /= keep_prob
        else:
            # last layer uses a softmax activation
            dZi = cache['A' + str(i)] - Y
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m
        weights['W' + str(i)] = weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = weights_copy['b' + str(i)] - alpha * dbi


def tanh(Y):
    """define the tanh activation function"""
    return np.tanh(Y)


def tanh_prime(Y):
    """define the derivative of the activation function tanh"""
    return 1 - Y ** 2
