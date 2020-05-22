#!/usr/bin/env python3
"""
Normalize
"""


def normalize(X, m, s):
    """function that normalizes a matrix"""
    X_norm = (X - m) / s
    return X_norm
