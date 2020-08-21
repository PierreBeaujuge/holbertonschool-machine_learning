#!/usr/bin/env python3
"""
1-gp.py
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

        # K: covariance kernel matrix of shape (m, n)
        K = (self.sigma_f ** 2) * np.exp(-0.5 * (1 / (self.l ** 2)) * dist_sq)
        # print("K.shape:", K.shape)

        return K

    def predict(self, X_s):
        """
        function that predicts the mean and standard deviation of points
        in a Gaussian process
        """

        # Call K
        K = self.K
        # Compute K_s in a call to kernel()
        K_s = self.kernel(self.X, X_s)
        # Compute K_ss in a call to kernel()
        K_ss = self.kernel(X_s, X_s)
        # Call Y
        Y = self.Y

        # The prediction follows a normal distribution completely
        # described by the mean "mu" and the covariance "sigma**2"

        # Compute the mean "mu"
        K_inv = np.linalg.inv(K)
        mu_s = np.matmul(np.matmul(K_s.T, K_inv), Y).reshape(-1)
        # Compute the covariance matrix "cov_s"
        cov_s = K_ss - np.matmul(np.matmul(K_s.T, K_inv), K_s)
        # Infer the standard deviation "sigma"
        sigma = np.diag(cov_s)

        return mu_s, sigma

    def update(self, X_new, Y_new):
        """function that updates a Gaussian process"""

        # Add the new sample point
        # print("X_prev:", self.X)
        # print("Y_prev:", self.Y)
        self.X = np.concatenate((self.X, X_new[..., np.newaxis]), axis=0)
        self.Y = np.concatenate((self.Y, Y_new[..., np.newaxis]), axis=0)
        # print("X_new:", self.X)
        # print("X_new:", self.Y)

        # Add the new function value
        # print("K_prev:", self.K)
        self.K = self.kernel(self.X, self.X)
        # print("K_new:", self.K)
