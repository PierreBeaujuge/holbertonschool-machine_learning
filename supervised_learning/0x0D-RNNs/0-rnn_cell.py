#!/usr/bin/env python3
"""
0-rnn_cell.py
"""
import numpy as np


class RNNCell:
    """define the class RNNCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""

        # Compute the concatenated cell input (given h_prev and x_t)
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # note: cell_input shape (m, i + h), Wh shape (i + h, h)

        # Compute h_next, the new hidden state of the cell
        h_next = np.tanh(np.matmul(cell_input, self.Wh) + self.bh)
        # note: h_next shape (m, h)

        # Compute the cell output (given h_next ->
        # i.e. taking into account the new hidden state of the cell)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))
