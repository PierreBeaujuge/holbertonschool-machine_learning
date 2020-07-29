#!/usr/bin/env python3
"""
0-pca.py
"""
import numpy as np


def pca(X, var=0.95):
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

    # Compute the SVD:
    U, S, Vt = np.linalg.svd(X)
    # print(S)

    # Compute the commulated sum of the singular (s) values from S:
    sum_s = np.cumsum(S)
    # print(sum_s)
    # sum_s is a 1D array of this type: array([ 1,  3,  6, 10, 15, 21])

    # Infer 'r' (number of principal components to extract from W/V)
    # based on the 'var' treshold passed as argument to the method
    # Normalize sum_s:
    sum_s = sum_s / sum_s[-1]
    # print(sum_s)
    # Note: here np.where() returns an array of indices
    r = np.min(np.where(sum_s >= var))
    # print(r)

    # Compute Vr(= Wr):
    V = Vt.T
    Vr = V[..., :r + 1]

    return Vr
