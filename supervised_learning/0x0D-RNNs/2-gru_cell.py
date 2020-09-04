#!/usr/bin/env python3
"""
2-gru_cell.py
"""
import numpy as np


class GRUCell:
    """define the class GRUCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""

        # Compute the concatenated cell input (given h_prev and x_t)
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # note: cell_input shape (m, i + h), Wh shape (i + h, h)

        # Compute the update gate and reset gate components
        update_gate = self.sigmoid(np.matmul(cell_input, self.Wz) + self.bz)
        reset_gate = self.sigmoid(np.matmul(cell_input, self.Wr) + self.br)

        # Next, h_prev gets multiplied with reset_gate, and the whole
        # is concatenated with x_t
        updated_cell_input = np.concatenate((reset_gate * h_prev, x_t), axis=1)

        # Compute h_r, the new hidden state of the cell
        # after applying the reset_gate
        h_r = np.tanh(np.matmul(updated_cell_input, self.Wh) + self.bh)
        # note: h_r shape (m, h)

        # Compute h_next, the new hidden state of the cell after factoring in
        # the update_gate contribution
        h_next = update_gate * h_r + (1 - update_gate) * h_prev

        # Compute the cell output (given h_next ->
        # i.e. taking into account the new hidden state of the cell)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-Y))
