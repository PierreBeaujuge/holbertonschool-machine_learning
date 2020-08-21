#!/usr/bin/env python3
"""
0-gp.py
"""
import numpy as np


class GaussianProcess:
    """Class that instantiates a noiseless 1D Gaussian process"""

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """define and initialize variables and methods"""

        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(self.X, self.X)

    def kernel(self, X1, X2):
        """
        function that calculates the covariance kernel matrix
        between two matrices
        """

        # Composition of the constant kernel with the
        # radial basis function (RBF) kernel, which encodes
        # for smoothness of functions (i.e. similarity of
        # inputs in space corresponds to the similarity of outputs)

        # Two hyperparameters: signal variance (sigma_f**2) and lengthscale l
        # K: Constant * RBF kernel function

        # Compute "dist_sq" (helper to K)
        # X1: shape (m, 1), m points of 1 coordinate
        # X2: shape (n, 1), n points of 1 coordinate
        a = np.sum(X1 ** 2, axis=1, keepdims=True)
        b = np.sum(X2 ** 2, axis=1, keepdims=True)
        c = np.matmul(X1, X2.T)
        # Note: Ensure a and b are aligned with c: shape (m, n)
        # -> b should be a row vector for the subtraction with c
        dist_sq = a + b.reshape(1, -1) - 2 * c
        # print("dist_sq:", dist_sq)

        # K: covariance kernel matrix of shape (m, n)
        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)

        return K
