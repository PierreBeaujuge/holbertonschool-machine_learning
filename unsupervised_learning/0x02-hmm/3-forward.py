#!/usr/bin/env python3
"""
3-forward.py
"""
import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    function that performs the forward algorithm for a hidden markov model
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

    # Initialize F (equivalent to alpha): shape (N, T)
    # F1(z1) = alpha1(z1) = p(z1,x1) = p(z1)p(x1/z1)
    # p(z1): Initial
    # p(x1/z1): Extract from Emission probability matrix
    F = np.zeros((N, T))
    # Use Observation[0] (1st observation in the time series)
    # "Observation" contains the index of the observation
    # print("Observation:", Observation)
    # (this array corresponds to the time series of observations
    # over which one should iterate)
    # print("Observation[0]:", Observation[0])
    index = Observation[0]
    # Emission contains the probabilities of each observation
    # corresponding to every given state
    # print("Emission:", Emission, Emission.shape)
    # Extract column of observation probabilities at "index" from Emission
    # print("Emission[:, Observation[0]]:", Emission[:, Observation[0]],
    #       Emission[:, Observation[0]].shape)
    Emission_idx = Emission[:, index]
    # Multiply Initial.T (array (1, N)) by Emission_idx (vector (N,))
    # This is an element-wise multiplication
    # print("Initial:", Initial, Initial.shape)
    # print("Initial.T:", Initial.T, Initial.T.shape)
    F[:, 0] = Initial.T * Emission_idx
    # print("F[:, 0]:", F[:, 0], F[:, 0].shape)

    # Iterate over N and T to compose F (alpha)
    # F: shape (N, T), containing the forward path probabilities
    # Start by iterating over the time steps T (number of observations)
    for j in range(1, T):
        for i in range(N):
            # Compute Fk(zk) = alphak(zk) =
            # sum(over zk-1=1,...N)(p(xk/zk)p(zk/zk-1)Fk-1(zk-1)) for k=2,...T
            # F[i, j]: probability of being in hidden state i at time j
            # given the previous observations
            F[i, j] = np.sum(Emission[i, Observation[j]]
                             * Transition[:, i] * F[:, j - 1], axis=0)
    # Evaluate the likelihood of the obervations given the model from F
    # Sum over the N states of the last event/observation
    # print("F[:, T-1:]:", F[:, T-1:])
    P = np.sum(F[:, T-1:], axis=0)[0]
    # print(np.sum(F, axis=0), np.sum(F, axis=0).shape)

    return P, F
