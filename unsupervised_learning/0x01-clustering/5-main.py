#!/usr/bin/env python3

import numpy as np
pdf = __import__('5-pdf').pdf

if __name__ == '__main__':
    np.random.seed(0)
    m = np.array([12, 30, 10])
    S = np.array([[36, -30, 15], [-30, 100, -20], [15, -20, 25]])
    X = np.random.multivariate_normal(m, S, 10000)
    P = pdf(X, m, S)
    print(P)

    # np.random.seed(0)
    # m = np.array([50, 75])
    # S = np.array([[30, 10], [10, 15]])
    # X = np.random.multivariate_normal(m, S, 10000)
    # P = pdf(X, m, S)
    # print(P)
    # print(P.shape)
