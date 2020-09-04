#!/usr/bin/env python3
"""
4-deep_rnn.py
"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """function that performs forward propagation for a deep RNN"""

    # X: shape (t, m, i)
    t = X.shape[0]
    m = X.shape[1]
    # h_0: shape (l, m, h)
    h = h_0.shape[2]
    # rnn_cells is a list
    k = len(rnn_cells)

    # Initialize an array H that will take in h_next at every time step
    H = np.zeros((t + 1, k, m, h))
    # Initialize an array Y that will take in y at every time step
    Y = np.zeros((t, m, rnn_cells[k - 1].Wy.shape[1]))

    for i in range(t):
        for j in range(k):
            if i == 0:
                H[i, j] = h_0[j]
            if j == 0:
                # Here, h_prev = h_0, x_t = X[i] and y = Y[i]
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j], X[i])
            else:
                H[i + 1, j], Y[i] = rnn_cells[j].forward(H[i, j],
                                                         H[i + 1, j - 1])

    return H, Y
