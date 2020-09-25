#!/usr/bin/env python3
"""
1-self_attention.py
"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """class that instantiates a self-attention layer"""

    def __init__(self, units):
        """constructor"""
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """function that builds the self-attention layer"""

        W = self.W(s_prev)[:, tf.newaxis, :]
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))

        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum((weights * hidden_states), axis=1)

        return context, weights
