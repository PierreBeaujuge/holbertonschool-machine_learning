#!/usr/bin/env python3
"""
Gradient Descent with L2 Regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """function that updates the weights and biases of a nn using
    gradient descent with L2 regularization"""
    weights_copy = weights.copy()
    for i in range(L, 0, -1):
        m = Y.shape[1]
        if i != L:
            # all layers use a tanh activation, except last
            # introduce call to tanh_prime method
            dZi = np.multiply(np.matmul(
                weights_copy['W' + str(i + 1)].T, dZi
            ), tanh_prime(cache['A' + str(i)]))
        else:
            # last layer uses a softmax activation
            dZi = cache['A' + str(i)] - Y
        dWi = np.matmul(dZi, cache['A' + str(i - 1)].T) / m
        dbi = np.sum(dZi, axis=1, keepdims=True) / m
        # L2 regularization: W multiplied by (1 - alpha * lambtha / m)
        # (1 - alpha * lambtha / m) here arbitrarily named "l2"
        # term l2 is < 1 -> "Weight Decay"
        l2 = (1 - alpha * lambtha / m)
        weights['W' + str(i)] = l2 * weights_copy['W' + str(i)] - alpha * dWi
        weights['b' + str(i)] = l2 * weights_copy['b' + str(i)] - alpha * dbi


def tanh(Y):
    """define the tanh activation function"""
    return np.tanh(Y)


def tanh_prime(Y):
    """define the derivative of the activation function tanh"""
    return 1 - Y ** 2
