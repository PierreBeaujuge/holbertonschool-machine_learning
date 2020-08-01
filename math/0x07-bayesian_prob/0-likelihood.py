#!/usr/bin/env python3
"""
0-likelihood.py
"""
import numpy as np


def likelihood(x, n, P):
    """
    function that calculates the likelihood of obtaining the data
    given various hypothetical probabilities of developing severe side effects
    """

    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        err = "x must be an integer that is greater than or equal to 0"
        raise ValueError(err)
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not all(P >= 0) or not all(P <= 1):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Product of probabilities for the data:
    A = (P ** x) * ((1 - P) ** (n - x))
    # Factorials to be accounted for:
    B = np.math.factorial(x) * np.math.factorial(n - x) / np.math.factorial(n)
    L = A / B

    return L
