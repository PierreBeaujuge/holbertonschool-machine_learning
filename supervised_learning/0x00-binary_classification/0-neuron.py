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
        self.W = np.random.normal(loc=0.0, scale=1.0, size=nx).reshape(1, nx)
        self.b = 0
        self.A = 0
