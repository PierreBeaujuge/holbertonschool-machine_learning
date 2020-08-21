#!/usr/bin/env python3
"""
4-bayes_opt.py
"""
import numpy as np
from scipy.stats import norm
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Class that instantiates a Bayesian optimization
    on a noiseless 1D Gaussian process
    """

    def __init__(self, f, X_init, Y_init, bounds,
                 ac_samples, l=1, sigma_f=1, xsi=0.01, minimize=True):
        """define and initialize variables and methods"""

        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1],
                               num=ac_samples)[..., np.newaxis]
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """function that calculates the next best sample location"""

        # Compute mu and sigma in a call to predict() on gp
        mu, sigma = self.gp.predict(self.X_s)
        # print("mu:", mu, mu.shape)
        # print("sigma:", sigma, sigma.shape)

        # Note: sigma of shape (s,)
        Z = np.zeros(sigma.shape)
        if self.minimize is True:
            f_plus = np.min(self.gp.Y)
            Z_NUM = f_plus - mu - self.xsi
        else:
            f_plus = np.max(self.gp.Y)
            Z_NUM = mu - f_plus - self.xsi

        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                Z[i] = Z_NUM[i] / sigma[i]
            else:
                Z[i] = 0

        # Compute the Expected Improvement (EI)
        EI = np.zeros(sigma.shape)
        for i in range(sigma.shape[0]):
            if sigma[i] > 0:
                EI[i] = Z_NUM[i] * norm.cdf(Z[i]) + sigma[i] * norm.pdf(Z[i])
            else:
                EI[i] = 0
        X_next = self.X_s[np.argmax(EI)]

        # print("EI:", EI)
        # print("self.X_s:", self.X_s)
        return X_next, EI
