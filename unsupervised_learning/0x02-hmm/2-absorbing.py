#!/usr/bin/env python3
"""
2-absorbing.py
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


def absorbing(P):
    """function that determines if a markov chain is absorbing"""

    if not isinstance(P, np.ndarray) or P.ndim != 2:
        return None
    if P.shape[0] != P.shape[1]:
        return None
    # Ensure the sum of all probabilities is equal to 1
    # when summing along the rows of the transition matrix "P"
    n = P.shape[0]
    if not np.isclose(np.sum(P, axis=1), np.ones(n))[0]:
        return None

    # Markov chain is absorbing if (1) it has at least one absorbing state
    # i.e. the corresponding row in P has a 1 intercept with diag
    if np.all(np.diag(P) != 1):
        return False
    # If all states are absorbing (P == identity matrix)
    # then the Markov chain is necessarily absorbing
    if np.all(np.diag(P) == 1):
        return True

    # Markov chain is absorbing if (2) it is possible to go from each
    # non-absorbing state to at least one absorbing state in a finite
    # number of steps
    # Note: it is common practice to compute the fundamental matrix of P
    # to demonstrate that any given transient (non-absorbing) state can
    # lead to an absorbing state in a finite number of steps

    # Assuming P is already in its standard form
    # Extract the sub-array R
    for i in range(n):
        if np.any(P[i, :] == 1):
            continue
        break
    # print("i", i)
    II = P[:i, :i]
    Id = np.identity(n - i)
    R = P[i:, :i]
    Q = P[i:, i:]
    # print("I:", II)
    # print("R:", R, R.shape)
    # print("Q:", Q)
    try:
        F = np.linalg.inv(Id - Q)
        # print("F:", F, F.shape)
    except Exception:
        return False
    FR = np.matmul(F, R)
    # print("FR:", FR)
    # Infer the limiting matrix
    Pbar = np.zeros((n, n))
    Pbar[:i, :i] = P[:i, :i]
    Pbar[i:, :i] = FR
    # print("Pbar:", Pbar)

    Qbar = Pbar[i:, i:]
    # print("Qbar:", Qbar)
    if np.all(Qbar == 0):
        return True
