#!/usr/bin/env python3
"""
Regularization Cost
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """function that calculates the cost of a nn with L2 regularization"""

    # function that calculates the Frobenius norm (when ord=None)
    # numpy.linalg.norm(x, ord=None, axis=None, keepdims=False)
    frobenius_norm = 0
    for weight in weights.values():
        frobenius_norm += np.linalg.norm(weight)
    cost += lambtha / (2 * m) * frobenius_norm
    return cost

    # failed attempts to calculate the Frobenius norm:
    # W = np.array([weights['W' + str(i)]
    #               for i in range(1, len(weights.keys()) + 1)])
    # frobenius_norm = np.sum([np.sum(W[i] ** 2) for i in range(len(W))])
    # frobenius_norm = np.sum([W[k][i][j] ** 2 for j in range(L)
    #                          for i in range(L - 1) for k in range(len(W))])
