#!/usr/bin/env python3
"""
6-baum_welch.py
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
    # T: Number of observations
    T = Observation.shape[0]

    # Initialize F (equivalent to alpha): shape (N, T)
    F = np.zeros((N, T))
    index = Observation[0]
    Emission_idx = Emission[:, index]
    F[:, 0] = Initial.T * Emission_idx

    # Iterate over N and T to compose F (alpha)
    # F: shape (N, T), containing the forward path probabilities
    for j in range(1, T):
        for i in range(N):
            # F[i, j]: probability of being in hidden state i at time j
            # given the previous observations
            F[i, j] = np.sum(Emission[i, Observation[j]]
                             * Transition[:, i] * F[:, j - 1], axis=0)
    # Evaluate the likelihood of the obervations given the model from F
    # Sum over the N states of the last event/observation
    P = np.sum(F[:, T-1:], axis=0)[0]

    return P, F


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
    # T: Number of observations
    T = Observation.shape[0]

    # Initialize B (equivalent to beta): shape (N, T)
    B = np.zeros((N, T))
    B[:, T - 1] = np.ones((N))

    # Iterate over N and T to compose B (beta)
    # B: shape (N, T), containing the backward path probabilities
    for j in range(T - 2, -1, -1):
        for i in range(N):
            # B[i, j]: the probability of generating the future observations
            # from hidden state i at time j
            B[i, j] = np.sum(B[:, j + 1] * Emission[:, Observation[j + 1]]
                             * Transition[i, :], axis=0)
    # Evaluate the likelihood of the observations given the model from B
    # Sum over the N states of the first event/observation
    P = np.sum(Initial.T * Emission[:, Observation[0]] * B[:, 0], axis=1)[0]

    return P, B


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    function that performs the Baum-Welch algorithm for a hidden markov model
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
    if not isinstance(Observations, np.ndarray) or Observations.ndim != 1:
        return None, None
    # Emission: shape (N, M), M: number of all possible observations
    if not isinstance(Emission, np.ndarray) or Emission.ndim != 2:
        return None, None
    if not np.sum(Emission, axis=1).all():
        return None, None
    if not np.isclose(np.sum(Emission, axis=1),
                      np.ones(Emission.shape[0])).all():
        return None, None
    # Iterations: number of times expectation-maximization should be performed
    if not isinstance(iterations, int) or iterations < 0:
        return None, None

    # N: Number of hidden states
    N = Initial.shape[0]
    # print("N:", N)
    # T: Number of observations
    T = Observations.shape[0]
    # print("T:", T)
    # M: Number of output states (observations)
    M = Emission.shape[1]

    a = Transition
    b = Emission
    # Make deep copies of "a" and "b" for early stop
    a_prev = np.copy(a)
    b_prev = np.copy(b)

    # for iteration in range(iterations):
    for iteration in range(1000):
        # print("iteration {}:".format(iteration))

        # Make calls to forward() and backward() to compute
        # F: alpha, aggregate helper variable; shape (N, T)
        # B: beta, aggregate helper variable; shape (N, T)
        PF, F = forward(Observations, b, a, Initial)
        PB, B = backward(Observations, b, a, Initial)

        # Compute X: ki, aggregate helper variable (Baum-Welch)
        X = np.zeros((N, N, T - 1))
        NUM = np.zeros((N, N, T - 1))
        for t in range(T - 1):
            # DEN = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    Fit = F[i, t]
                    aij = a[i, j]
                    bjt1 = b[j, Observations[t + 1]]
                    Bjt1 = B[j, t + 1]
                    NUM[i, j, t] = Fit * aij * bjt1 * Bjt1
        # print("NUM:", NUM)
        DEN = np.sum(NUM, axis=(0, 1))
        X = NUM / DEN
        # print("X:", X, X.shape)
        # print("X:", X.shape)

        # Compute G: gamma, aggregate helper variable (Viterbi)
        # G_bis = np.sum(X, axis=1)
        # print("G_bis:", G_bis[:,-2:], G_bis.shape)
        G = np.zeros((N, T))
        NUM = np.zeros((N, T))
        for t in range(T):
            for i in range(N):
                Fit = F[i, t]
                Bit = B[i, t]
                NUM[i, t] = Fit * Bit
        # print("NUM:", NUM)
        DEN = np.sum(NUM, axis=0)
        # print("DEN:", DEN)
        G = NUM / DEN
        # print("G:", G[:,-2:], G.shape)

        # Update the Transition matrix "a"
        a = np.sum(X, axis=2) / np.sum(G[:, :T - 1], axis=1)[..., np.newaxis]
        # print("a:", a, a.shape)

        # Update the Emission matrix "b"
        DEN = np.sum(G, axis=1)
        # print("DEN:", DEN, DEN.shape)
        # print("b.shape:", b.shape)
        NUM = np.zeros((N, M))
        for k in range(M):
            NUM[:, k] = np.sum(G[:, Observations == k], axis=1)
        b = NUM / DEN[..., np.newaxis]
        # print("b:", b, b.shape)

        # Early stopping; exit condition on "a" and "b"
        if np.all(np.isclose(a, a_prev)) or np.all(np.isclose(a, a_prev)):
            return a, b

        # Make deep copies of "a" and "b" (for early stop)
        a_prev = np.copy(a)
        b_prev = np.copy(b)

    return a, b
