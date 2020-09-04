#!/usr/bin/env python3
"""
3-lstm_cell.py
"""
import numpy as np


class LSTMCell:
    """define the class LSTMCell"""

    def __init__(self, i, h, o):
        """constructor"""

        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """function that performs forward propagation for one time step"""

        # Compute the concatenated cell input (given h_prev and x_t)
        cell_input = np.concatenate((h_prev, x_t), axis=1)
        # note: cell_input shape (m, i + h), Wh shape (i + h, h)

        # Compute the forget gate and update gate components
        forget_gate = self.sigmoid(np.matmul(cell_input, self.Wf) + self.bf)
        update_gate = self.sigmoid(np.matmul(cell_input, self.Wu) + self.bu)

        # Compute c_inter, the intermediate cell state
        c_inter = np.tanh(np.matmul(cell_input, self.Wc) + self.bc)
        # note: c_inter shape (m, h)

        # Compute c_next, the cell state of the next cell
        # given c_prev and c_inter
        c_next = c_prev * forget_gate + update_gate * c_inter

        # Compute the output gate compoenent
        output_gate = self.sigmoid(np.matmul(cell_input, self.Wo) + self.bo)

        # Compute h_next, the new state of the cell after factoring in
        # the output_gate contribution
        h_next = output_gate * np.tanh(c_next)

        # Compute the cell output (given c_next ->
        # i.e. taking into account the new cell state)
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    def softmax(self, Y):
        """define the softmax activation function"""
        return np.exp(Y) / (np.sum(np.exp(Y), axis=1, keepdims=True))

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-Y))
