#!/usr/bin/env python3
"""
1-q_init.py
"""
import numpy as np
import gym


def q_init(env):
    """
    function that initializes the Q-table
    """

    action_space_size = env.action_space.n
    state_space_size = env.observation_space.n

    q_table = np.zeros(shape=(state_space_size, action_space_size))

    return q_table
