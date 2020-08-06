#!/usr/bin/env python3
"""
1-kmeans.py
"""
import numpy as np


def initialize(X, k):
    """function that initializes cluster centroids for K-means"""

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape
    # print(X.shape)
    # print(X)

    if not isinstance(k, int) or k <= 0 or k > n:
        return None

    # Sample k centroids from a random.uniform distribution;
    # output is an array of coordinates
    C = np.random.uniform(low=np.min(X, axis=0),
                          high=np.max(X, axis=0),
                          size=(k, d))
    return C


def kmeans(X, k, iterations=1000):
    """function that performs K-means clustering on a dataset"""

    # Initialize the cluster centroids (C <- centroid "means")
    C = initialize(X, k)

    if C is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    # # Initialize the cost/distortion function;
    # # defined as J = sum/n(sum/k(r(ij)*||x(i) - c(j)||**2))
    # J = np.inf

    # Iterate over iterations
    for iteration in range(iterations):
        # print("iteration:", iteration)

        # Maintain a deep copy of C
        # C_prev = np.array([x for x in C])
        # Another alternative (removes for loop):
        C_prev = np.copy(C)

        # OPTION 1: FOR LOOPS

        # Initialize the array of pairwise data point-centroid
        # distances with zeros
        # dist = np.zeros((n, k))

        # for i in range(n):
        #     for j in range(k):
        #         dist[i, j] = np.linalg.norm(X[i, ...] - C[j, ...])
        # Note: squared distances can alternatively be inferred
        # directtly from the inner product of (X - C) with itself
        # dist[i, j] = np.inner(X[i,:]-C[j,:], X[i,:]-C[j,:])
        # print("dist:", dist)
        # Squared distances from "dist":
        # print("dist ** 2:", dist ** 2)

        # OPTION 2: VECTORIZATION

        # Convert X into an array suitable for vectorization
        Xv = np.repeat(X, k, axis=0)
        # print("Xv:", Xv)
        # print("Xv.shape:", Xv.shape)
        Xv = Xv.reshape(n, k, d)
        # print("Xv:", Xv)
        # print("Xv.shape:", Xv.shape)

        # Convert C into an array suitable for vectorization
        Cv = np.tile(C, (n, 1))
        # print("Cv:", Cv)
        # print("Cv.shape:", Cv.shape)
        Cv = Cv.reshape(n, k, d)
        # print("Cv:", Cv)
        # print("Cv.shape:", Cv.shape)

        # Compute the "dist" matrix of euclidean distances between
        # data points and centroids; shape (n, k)
        dist = np.linalg.norm(Xv - Cv, axis=2)

        # Assign each point of the dataset to a centroid:
        # Evaluate argmin(dist**2) for comparison with k
        # r(ij) = 1 if argmin(dist**2) == j
        # -> point i assigned to centroid k
        # otherwise r(ij) = 0 -> ignore point i wrt centroid k
        clss = np.argmin(dist ** 2, axis=1)
        # print("centroid indices:", clss)
        # print("clss.shape:", clss.shape)
        # Note: here, clss is a 1D array of the unique centroid index
        # to which each point in the dataset as been assigned (closest to);
        # the indices array is used in place of r(ij) in J evaluations

        # OPTION 1: EXIT CONDITION BASED ON J_prev == J

        # # Make a copy of the previous J value & reinitialize J
        # J_prev = J
        # # J = 0

        # # Update J (summing over the n data points),
        # # based on the (shortest) distances inferred from "indices"
        # # From "for" loop:
        # # for i in range(n):
        # #     J += (dist[i, clss[i]] ** 2)
        # # From vectorization:
        # J = np.sum(dist[..., clss] ** 2)
        # # Normalize J to the number of data points to
        # # reduce the computational cost (optional)
        # J /= n
        # # print("J:", J)

        # if J == J_prev:
        #     # print("last iteration:", iteration)
        #     return C, clss

        # Move the cluster centroids to the center (mean) of
        # the refined cluster by updating C (centroid coordinates)
        for j in range(k):
            # Infer the array of data point indices that correspond
            # to each assigned cluster centroid
            indices = np.where(clss == j)[0]
            # print("indices:", indices)
            if len(indices) == 0:
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[indices], axis=0)

        # OPTION 2: EXIT CONDITION BASED ON C == C_prev

        if (C == C_prev).all():
            # print("last iteration:", iteration)
            return C, clss

    # Update clss before returning C, clss
    Cv = np.tile(C, (n, 1))
    Cv = Cv.reshape(n, k, d)
    dist = np.linalg.norm(Xv - Cv, axis=2)
    clss = np.argmin(dist ** 2, axis=1)

    return C, clss
