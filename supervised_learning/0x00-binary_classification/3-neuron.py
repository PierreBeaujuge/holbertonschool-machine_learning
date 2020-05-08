#!/usr/bin/env python3
"""
Binary Classification
"""
import numpy as np


class Neuron:
    """
    define the Neuron class
    """

    def __init__(self, nx):
        """initialize variables and methods"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.normal(loc=0.0, scale=1.0, size=nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation function"""
        Z = np.matmul(self.W, X) + self.b
        self.__A = self.sigmoid(Z)
        return self.A

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * Y))

    def cost(self, Y, A):
        """defnine the cost function"""
        # classification pb: use the cross-entropy function
        m = Y.shape[1]
        return (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
        # also works (use np.mean):
        # return -np.mean(Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))
