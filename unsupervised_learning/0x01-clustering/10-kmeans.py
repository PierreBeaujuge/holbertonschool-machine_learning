#!/usr/bin/env python3
"""
10-kmeans.py
"""
import sklearn.cluster


def kmeans(X, k):
    """
    function that performs K-means on a dataset
    """

    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = kmeans.cluster_centers_
    clss = kmeans.labels_

    return C, clss
