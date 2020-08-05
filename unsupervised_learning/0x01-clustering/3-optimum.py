#!/usr/bin/env python3
"""
3-optimum.py
"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def initialize(X, k):
    """function that initializes cluster centroids for K-means"""

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape
    # print(X.shape)
    # print(X)

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(k, int) or k <= 0 or k >= n:
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

    if type(C) is None:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    # Initialize the cost/distortion function;
    # defined as J = sum/n(sum/k(r(ij)*||x(i) - c(j)||**2))
    J = np.inf

    # Iterate over iterations
    for iteration in range(iterations):
        # print("iteration:", iteration)

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

        # Make a copy of the previous J value & reinitialize J
        J_prev = J
        # J = 0

        # Update J (summing over the n data points),
        # based on the (shortest) distances inferred from "indices"
        # From "for" loop:
        # for i in range(n):
        #     J += dist[i, clss[i]]
        # From vectorization:
        J = np.sum(dist[..., clss])
        # Normalize J to the number of data points to
        # reduce the computational cost (optional)
        J /= n
        # print("J:", J)

        if np.any(J == J_prev):
            return C, clss

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

    return C, clss


def variance(X, C):
    """
    function that calculates the total intra-cluster variance for a data set
    """

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(C, np.ndarray) or C.ndim != 2:
        return None

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    # Number of cluster centroids of shape (k, d)
    k = C.shape[0]

    if k > n:
        return None
    if C.shape[1] != d:
        return None

    # OPTION 1: FOR LOOPS

    # Initialize the array of pairwise data point-centroid
    # distances with zeros
    # dist = np.zeros((n, k))

    # Compute the "dist" matrix of euclidean distances between
    # data points and centroids
    # for i in range(n):
    #     for j in range(k):
    #         dist[i, j] = np.linalg.norm(X[i, ...] - C[j, ...])

    # OPTION 2: VECTORIZATION

    # Convert X into an array suitable for vectorization
    Xv = np.repeat(X, k, axis=0)
    Xv = Xv.reshape(n, k, d)

    # Convert C into an array suitable for vectorization
    Cv = np.tile(C, (n, 1))
    Cv = Cv.reshape(n, k, d)

    # Compute the "dist" matrix of euclidean distances between
    # data points and centroids; shape (n, k)
    dist = np.linalg.norm(Xv - Cv, axis=2)

    # Determine the centroid to which each data point relates to
    # clss = np.argmin(dist, axis=1)
    # Compute the 1D array of shortest data point-centroid
    # squared distances
    short_dist = np.min(dist ** 2, axis=1)
    # print("short_dist:", short_dist)
    # print("short_dist.shape:", short_dist.shape)

    # Sum of "short_dist" over the n data points == definition of
    # overall intra-cluster variance for the dataset
    # Evaluate the variance of the corresponding array
    var = np.sum(short_dist)

    return var


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """function that tests for the optimum number of clusters by variance"""

    # n: number of dada points
    # d: dimension of each data point
    n, d = X.shape

    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None
    if kmax is None:
        kmax = n
    if not isinstance(kmin, int) or kmin <= 0 or n <= kmin:
        return None, None
    if not isinstance(kmax, int) or kmax <= 0 or n < kmax:
        return None, None
    if kmin >= kmax:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    # Initialize list of tuples (C, clss)
    results = []
    # Initialize list of total intra-cluster variances
    variances = []
    # Initialize list of difference in variance from
    # the smallest cluster size for each cluster size
    d_vars = []

    # Iterate over the number of clusters under consideration
    for k in range(kmin, kmax+1):

        # Compute the cluster centroids C (means; coordinates and
        # the 1D array of data point-centroid assignement in a call to kmeans()
        C, clss = kmeans(X, k, iterations)
        results.append((C, clss))

        # Compute the corresponding total intra-cluster variance
        var = variance(X, C)
        variances.append(var)

    for var in variances:
        d_vars.append(np.abs(variances[0] - var))

    return results, d_vars
