#!/usr/bin/env python3
"""
5-bi_forward.py
"""
import numpy as np


class BidirectionalCell:
    """define the class BidirectionalCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h + h, o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """function that performs forward propagation for one time step"""

        # Compute the concatenated cell input (given h_prev and x_t)
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # note: cell_input shape (m, i + h), Wh shape (i + h, h)

        # Compute h_next, the new hidden state of the cell
        h_next = np.tanh(np.matmul(cell_input, self.Whf) + self.bhf)
        # note: h_next shape (m, h)

        return h_next
