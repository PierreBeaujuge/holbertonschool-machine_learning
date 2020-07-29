#!/usr/bin/env python3
"""
0-pca.py
"""
import numpy as np


def pca(X, ndim):
    """function that performs principal components analysis on a dataset"""

    # Useful resource: https://www.youtube.com/watch?v=NUn6WeFM5cM

    # PCA defined as the eigendecomposition of the covariance matrix dot(X.T,X)
    # -> returns a set of eigenvectors (W:loadings) and eigenvalues (lambda)
    # Each column of W is a principal component; columns ordered by how large
    # their corresponding eigenvalues are; i.e. principal component
    # corresponding to the largest eigenvalue is always going to be the first
    # one, Etc.; r: number of principal components to extract -> truncate W
    # -> extract the first r columns of W
    # T:scores, defined as T = dot(X,W) -> Tr = dot(X,Wr)
    # -> Tr new representation of X (in the W space)

    # Better way of computing W: Singular Value Decomposition (SVD)
    # X = dot(U,S,V.T); U: right singular vector, V:left singular vector
    # S: singular values on the diagonal (proportional to eigenvalues; ordered)
    # Importantly (it can be demonstrated that): V = W
    # T = dot(X,W) && X = dot(U,S,V.T) -> dot(X,V) = dot(U,S,V.T,V)
    # since dot(U.T,U) = I and dot(V.T,V) = I -> dot(X,V) = dot(U,S)
    # since V = W -> dot(X,W) = dot(U,S) -> T = dot(U,S) -> Tr = dot(Ur,Sr)

    # Ensure that all dimensions have a mean of 0 across all data points
    X = X - np.mean(X, axis=0)

    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)
    # print(S)

    # Compute Tr, the transformed matrix X:
    # print(U[..., :ndim])
    # print(np.diag(S[..., :ndim]))
    Tr = np.matmul(U[..., :ndim], np.diag(S[..., :ndim]))

    return Tr
