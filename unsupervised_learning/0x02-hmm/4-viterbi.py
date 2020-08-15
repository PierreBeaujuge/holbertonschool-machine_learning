#!/usr/bin/env python3
"""
4-viterbi.py
"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """function that that calculates the most likely sequence of hidden states
    for a hidden markov model"""

    # Initial: shape (N, 1), N: number of hidden states
    if not isinstance(Initial, np.ndarray) or Initial.ndim != 2:
        return None, None
    if Initial.shape[1] != 1:
        return None, None
    if not np.isclose(np.sum(Initial, axis=0), [1])[0]:
        return None, None
    # Transition: shape (N, N)
    if not isinstance(Transition, np.ndarray) or Transition.ndim != 2:
        return None, None
    if Transition.shape[0] != Initial.shape[0]:
        return None, None
    if Transition.shape[1] != Initial.shape[0]:
        return None, None
    if not np.isclose(np.sum(Transition, axis=1),
                      np.ones(Initial.shape[0])).all():
        return None, None
    # Observation: shape (T,), T: number of observations
    if not isinstance(Observation, np.ndarray) or Observation.ndim != 1:
        return None, None
    # Emission: shape (N, M), M: number of all possible observations
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None

    # N: Number of hidden states
    N = Initial.shape[0]
    # print("N:", N)
    # T: Number of observations
    T = Observation.shape[0]
    # print("T:", T)

    # Initialize an array V (equivalent to alpha): shape (N, T)
    # V1(z1) = alpha1(z1) = p(z1,x1) = p(z1)p(x1/z1)
    # p(z1): Initial
    # p(x1/z1): Extract from Emission probability matrix
    V = np.zeros((N, T))
    V[:, 0] = Initial.T * Emission[:, Observation[0]]
    # print("V[:, 0]:", V[:, 0], V[:, 0].shape)

    # Initialize an array B, keeping track of the possible state sequences
    B = np.zeros((N, T))

    for j in range(1, T):
        for i in range(N):
            # Apply the Forward algorithm to compose V (recursive/dynamic)
            # Compute Vk(zk) = alphak(zk) =
            # sum(over zk-1=1,...N)(p(xk/zk)p(zk/zk-1)Vk-1(zk-1)) for k=2,...T
            # V[i, j]: probability of being in hidden state i at time j
            # given the previous observations
            temp = Emission[i, Observation[j]] * Transition[:, i] * V[:, j - 1]
            # print("temp:", temp)
            V[i, j] = np.max(temp, axis=0)
            # print("V at {}:".format(i), V)
            B[i, j] = np.argmax(temp, axis=0)
            # print("B at {}:".format(i), B)
    # Extract the forward path probabilities for the last observation
    # and from this column vector, extract the max value <- max probability
    # -> infer the most likely path sequence; P: corresponding probability
    P = np.max(V[:, T - 1])
    # Infer the index of the corresponding state (last hidden state)
    # (most likely hidden state at T - 1)
    S = np.argmax(V[:, T - 1])
    # Add S to a "path" array
    path = [S]
    # Iterate over the remaining time steps, and then reverse path
    for j in range(T - 1, 0, -1):
        S = int(B[S, j])
        path.append(S)
    path = path[:: -1]
    # print("path:", len(path))

    return path, P
