#!/usr/bin/env python3
"""
1-regular.py
"""
import numpy as np


def markov_chain(P, s, t=1):
    """function that determines the probability of a markov chain being
    in a particular state after a specified number of iterations"""

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    if not isinstance(s, np.ndarray) or s.ndim != 2:
        return None
    if s.shape[0] != 1 or s.shape[1] != P.shape[1]:
        return None
    if not isinstance(t, int) or t < 0:
        return None
    # Ensure the sum of all probabilities is equal to 1
    # when summing along the rows of the transition matrix "P"
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None
    # Ensure the sum of all probabilities is equal to 1
    # when summing over the 1D array of starting probabilities "s"
    if not np.isclose(np.sum(s, axis=0), [1])[0]:
        return None

    # Perform t np.matmul() operations
    Pt = np.linalg.matrix_power(P, t)
    # Perform dot product of a and Pt
    Ps = np.matmul(s, Pt)

    return Ps


def regular(P):
    """function that determines the steady state probabilities
    of a regular markov chain"""

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    # Ensure the sum of all probabilities is equal to 1
    # when summing along the rows of the transition matrix "P"
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None

    # Initialize a 1D array of starting probabilities "s"
    # such that each entry has the same probability value
    # with equiprobability across states
    s = np.full(n, (1 / n))[np.newaxis, ...]

    # OPTION #1: iterate over k/powers (less efficient)

    # k = 0
    # while True:
    #     k += 1
    #     Pk = np.linalg.matrix_power(P, k)
    #     Ps = np.matmul(s, Pk)
    #     s = np.matmul(Ps, np.linalg.inv(P))
    #     # Markov chain must be regular, i.e. P and successive
    #     # powers of P must contain only positive entries
    #     if np.any(Pk <= 0):
    #         return None
    #     # Exit condition: when "s = sP"
    #     # (steady state reached)
    #     if np.all(Ps == s):
    #         return s

    # OPTION #2: incremental matmul() operations

    # Make a deep copy of P
    Pk = np.copy(P)
    # Initialize s_prev as s
    s_prev = s

    while True:
        Pk = np.matmul(Pk, P)
        if np.any(Pk <= 0):
            return None
        # Reinitialize "s" with "Ps"
        # (since Ps = np.matmul(s, Pk))
        s = np.matmul(s, P)
        # Exit condition: when "s = sP"
        # (steady state reached)
        if np.all(s_prev == s):
            return s
        # Save s as s_prev
        s_prev = s
