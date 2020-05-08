#!/usr/bin/env python3
"""
Binary Classification
"""
import numpy as np


class NeuralNetwork:
    """
    define the NeuralNetwork class
    """

    def __init__(self, nx, nodes):
        """initialize variables and methods"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')
        self.nx = nx
        self.nodes = nodes
        self.W1 = np.random.normal(loc=0.0, scale=1.0, size=(nodes, nx))
        self.b1 = np.zeros(nodes).reshape(nodes, 1)
        self.A1 = 0
        self.W2 = np.random.normal(
            loc=0.0, scale=1.0, size=nodes).reshape(1, nodes)
        self.b2 = 0
        self.A2 = 0
