#!/usr/bin/env python3
"""
3-q_learning.py
"""
import numpy as np
import time


def play(env, Q, max_steps=100):
    """
    function that has the trained agent play an episode
    """

    state = env.reset()
    done = False
    time.sleep(1)

    for step in range(max_steps):

        env.render()
        time.sleep(3.0)

        # Infer next action from current state (outside training -> q-table)
        action = np.argmax(Q[state, :])
        # Predict the next state based on action
        new_state, reward, done, info = env.step(action)

        # Handle episode termination:
        # if new_state is b'H' or b'G' --> episode ends (done == True)
        # with reward +1 if b'G' and -1 if b'H'
        if done is True:
            env.render()
            break

        # Update state
        state = new_state

    env.close()

    return reward
