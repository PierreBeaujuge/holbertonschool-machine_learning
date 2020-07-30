#!/usr/bin/env python3
"""
3-entropy.py
"""
import numpy as np


def HP(Di, beta):
    """
    function that calculates the Shannon entropy and
    P affinities relative to a data point
    """

    # Compute the affinities of the points
    A = np.exp(-Di * beta)
    B = np.sum(np.exp(-Di * beta), axis=0)
    Pi = A / B

    # Compute the Shannon entropy of the points
    Hi = -np.sum(Pi * np.log2(Pi))

    return Hi, Pi
