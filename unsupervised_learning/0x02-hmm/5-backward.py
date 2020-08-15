#!/usr/bin/env python3
"""
5-backward.py
"""
import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """
    function that performs the backward algorithm for a hidden markov model
    """

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

    # Initialize B (equivalent to beta): shape (N, T)
    # B1(z1) = beta1(z1) = 1
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    # Iterate over N and T to compose B (beta)
    # B: shape (N, T), containing the backward path probabilities
    # Start by iterating over the time steps T (number of observations)
    # but on the reverse path this time
    for j in range(T - 2, -1, -1):
        for i in range(N):
            # Compute Bk(zk) = betak(zk) =
            # sum(over zk-1=1,...N)(Bk+1(zk+1)p(xk+1/zk+1)p(zk+1/zk))
            # for k=1,...T - 1
            # B[i, j]: the probability of generating the future observations
            # from hidden state i at time j
            B[i, j] = np.sum(B[:, j + 1] * Emission[:, Observation[j + 1]]
                             * Transition[i, :], axis=0)
    # Evaluate the likelihood of the observations given the model from B
    # Sum over the N states of the first event/observation
    # print("Initial.T * Emission[:, Observation[0]] * B[:, 0]:",
    #       Initial.T * Emission[:, Observation[0]] * B[:, 0])
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]
    # print(np.sum(Initial.T *
    #              Emission[:, Observation[0]] *
    #              B[:, 0], axis=1)[0],
    #       np.sum(Initial.T *
    #              Emission[:, Observation[0]] *
    #              B[:, 0], axis=1).shape)

    return P, B
