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

        s_prev_with_time_axis = tf.expand_dims(s_prev, 1)

        W = self.W(s_prev_with_time_axis)
        U = self.U(hidden_states)
        score = self.V(tf.nn.tanh(W + U))

        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = tf.reduce_sum((attention_weights * hidden_states),
                                       axis=1)

        return context_vector, attention_weights
