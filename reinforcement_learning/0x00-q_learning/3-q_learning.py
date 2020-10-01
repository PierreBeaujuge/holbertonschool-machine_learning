#!/usr/bin/env python3
"""
3-q_learning.py
"""
import numpy as np
epsilon_greedy = __import__('2-epsilon_greedy').epsilon_greedy


def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    function that performs Q-learning
    """

    total_rewards = []

    # Q-learning algorithm
    for episode in range(episodes):
        state = env.reset()
        done = False

        for step in range(max_steps):

            # Exploration-exploitation trade-off
            action = epsilon_greedy(Q, state, epsilon)

            new_state, reward, done, info = env.step(action)

            # Check if new_state is b'H' (hole) <-- -1 reward
            map_size = env.desc.shape[0]
            new_state_on_map = env.desc[int(np.floor(new_state / map_size)),
                                        new_state % map_size]
            if new_state_on_map == b'H':
                reward = -1.0

            # Print the new_state/reward/done pattern
            # print("Episode:", episode)
            # print("new_state_on_map:", new_state_on_map)
            # print("new_state, reward, done:", new_state, reward, done)

            # Update q-table for Q(s,a)
            Q[state, action] = ((1 - alpha) * Q[state, action] + alpha *
                                (reward + gamma * np.max(Q[new_state, :])))

            # Update state
            state = new_state

            # Handle episode termination:
            # if new_state is b'H' or b'G' --> episode ends (done == True)
            # with reward +1 if b'G' and -1 if b'H'
            if done is True:
                break

        # Update exploration rate decay
        max_epsilon = 1
        epsilon = (min_epsilon + (max_epsilon - min_epsilon) *
                   np.exp(-epsilon_decay * episode))

        # Update total_rewards with episode reward
        total_rewards.append(reward)

    return Q, total_rewards
