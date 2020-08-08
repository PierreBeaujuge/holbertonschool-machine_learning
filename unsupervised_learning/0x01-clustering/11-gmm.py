#!/usr/bin/env python3
"""
11-gmm.py
"""
import sklearn.mixture


def gmm(X, k):
    """
    function that calculates a GMM from a dataset
    """

    GMM = sklearn.mixture.GaussianMixture(n_components=k)
    params = GMM.fit(X)
    pi = params.weights_
    m = params.means_
    S = params.covariances_
    clss = GMM.predict(X)
    bic = GMM.bic(X)

    return pi, m, S, clss, bic
