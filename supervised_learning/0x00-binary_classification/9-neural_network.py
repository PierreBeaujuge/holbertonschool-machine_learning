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
        self.__W1 = np.random.normal(loc=0.0, scale=1.0, size=(nodes, nx))
        self.__b1 = np.zeros(nodes).reshape(nodes, 1)
        self.__A1 = 0
        self.__W2 = np.random.normal(
            loc=0.0, scale=1.0, size=nodes).reshape(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """getter for W1"""
        return self.__W1

    @property
    def b1(self):
        """getter for b1"""
        return self.__b1

    @property
    def A1(self):
        """getter for A1"""
        return self.__A1

    @property
    def W2(self):
        """getter for W2"""
        return self.__W2

    @property
    def b2(self):
        """getter for b2"""
        return self.__b2

    @property
    def A2(self):
        """getter for A2"""
        return self.__A2
