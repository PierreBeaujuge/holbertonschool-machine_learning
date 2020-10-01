#!/usr/bin/env python3
"""
2-epsilon_greedy.py
"""
import numpy as np
import gym


def epsilon_greedy(Q, state, epsilon):
    """
    function that uses epsilon-greedy to determine the next action
    """

    # Exploration-exploitation trade-off
    exploration_rate_threshold = np.random.uniform(0, 1)
    if exploration_rate_threshold > epsilon:
        # Decide action based on Q (q-table)
        action = np.argmax(Q[state, :])
    else:
        # Sample randomly from the action space
        action = np.random.randint(Q.shape[1])

    return action
