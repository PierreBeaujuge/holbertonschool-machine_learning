#!/usr/bin/env python3
"""
multinormal.py
"""
import numpy as np


class MultiNormal:
    """
    Multivariate Normal distribution
    """

    def __init__(self, data):
        """constructor"""

        err_1 = "data must be a 2D numpy.ndarray"
        if not isinstance(data, np.ndarray):
            raise TypeError(err_1)
        if data.ndim != 2:
            raise TypeError(err_1)

        err_2 = "data must contain multiple data points"
        if data.shape[1] < 2:
            raise ValueError(err_2)

        self.mean, self.cov = self.mean_cov(data)

    @staticmethod
    def mean_cov(X):
        """
        function that calculates the mean and covariance matrix of a data set
        """

        d = X.shape[0]
        n = X.shape[1]

        mean = np.mean(X, axis=1)
        mean = mean[..., np.newaxis]

        X = X - mean
        cov = np.matmul(X, X.T) / (n - 1)

        return mean, cov

    def pdf(self, x):
        """
        function that calculates the PDF at a data point
        """

        err_1 = "x must be a numpy.ndarray"
        if not isinstance(x, np.ndarray):
            raise TypeError(err_1)

        d = self.cov.shape[0]
        err_2 = "x must have the shape ({}, 1)".format(d)
        if x.ndim != 2:
            raise ValueError(err_2)
        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError(err_2)

        # A = 1.0 / ((2 * np.pi) ** (d / 2) * np.linalg.det(self.cov) ** 0.5)
        A = 1.0 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov))
        B = np.exp(-0.5 * np.linalg.multi_dot([(x - self.mean).T,
                                               np.linalg.inv(self.cov),
                                               (x - self.mean)]))
        PDF = A * B

        return float(PDF)
