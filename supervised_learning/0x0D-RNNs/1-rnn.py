#!/usr/bin/env python3
"""
1-rnn.py
"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """function that performs forward propagation for the RNN"""

    # X: shape (t, m, i)
    t = X.shape[0]
    m = X.shape[1]
    # h_0: shape (m, h)
    h = h_0.shape[1]

    # Initialize an array H that will take in h_next at every time step
    H = np.zeros((t + 1, m, h))
    # Initialize an array Y that will take in y at every time step
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    for i in range(t):
        if i == 0:
            H[i] = h_0
        # Here, h_prev = h_0, x_t = X[i] and y = Y[i]
        H[i + 1], Y[i] = rnn_cell.forward(H[i], X[i])

    return H, Y
