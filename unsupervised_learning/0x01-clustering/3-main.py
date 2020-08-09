#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
optimum_k = __import__('3-optimum').optimum_k

if __name__ == "__main__":

    np.random.seed(0)
    means = np.random.uniform(0, 100, (3, 2))
    a = np.random.multivariate_normal(means[0], 10 * np.eye(2), size=10)
    b = np.random.multivariate_normal(means[1], 10 * np.eye(2), size=10)
    c = np.random.multivariate_normal(means[2], 10 * np.eye(2), size=10)
    X = np.concatenate((a, b, c), axis=0)
    # print(X)
    np.random.shuffle(X)
    res, v = optimum_k(X)
    print(res)
    # print("res:", res)
    # print("v:", v)
    print(np.round(v, 5))

    # np.random.seed(0)
    # a = np.random.multivariate_normal([30, 40], [[16, 0], [0, 16]], size=50)
    # b = np.random.multivariate_normal([10, 25], [[16, 0], [0, 16]], size=50)
    # c = np.random.multivariate_normal([40, 20], [[16, 0], [0, 16]], size=50)
    # d = np.random.multivariate_normal([60, 30], [[16, 0], [0, 16]], size=50)
    # e = np.random.multivariate_normal([20, 70], [[16, 0], [0, 16]], size=50)
    # X = np.concatenate((a, b, c, d, e), axis=0)
    # np.random.shuffle(X)
    # # print(X)

    # results, d_vars = optimum_k(X, kmax=10)
    # print(results)
    # print(np.round(d_vars, 5))
    # plt.scatter(list(range(1, 11)), d_vars)
    # plt.xlabel('Clusters')
    # plt.ylabel('Delta Variance')
    # plt.title('Optimizing K-means')
    # plt.show()
