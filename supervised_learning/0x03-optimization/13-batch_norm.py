#!/usr/bin/env python3
"""
Batch Normalization
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """function that normalizes an unactivated output by batch normalization"""
    m, stddev = normalization_constants(Z)
    s = stddev ** 2
    Z_norm = (Z - m) / np.sqrt(s + epsilon)
    Z_b_norm = gamma * Z_norm + beta
    return Z_b_norm


def normalization_constants(X):
    """function that calculates the normalization constants of a matrix"""
    m = X.shape[0]
    mean = np.sum(X, axis=0) / m
    stddev = np.sqrt(np.sum((X - mean) ** 2, axis=0) / m)
    return mean, stddev
