#!/usr/bin/env python3
"""
6-multi_head_attention.py
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """class that instantiates a multi-head attention block"""

    def __init__(self, dm, h):
        """constructor"""
        super(MultiHeadAttention, self).__init__()
        # Number of heads in the block
        self.h = h
        # Depth of the model
        self.dm = dm
        # Depth of each head
        self.depth = int(dm / h)

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """function that splits data over the last axis of any given array"""

        # Split the last axis into (h, depth) (from (dm,))
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        # Transpose to expected shape (batch_size, num_heads, seq_len, depth)
        x = tf.transpose(x, perm=[0, 2, 1, 3])

        return x

    def call(self, Q, K, V, mask):
        """function that builds the multi-head attention block"""

        batch_size = tf.shape(Q)[0]

        # Pass Q, K, V to their respective Dense layer
        Q = self.Wq(Q)  # (batch_size, seq_len_q, dm)
        K = self.Wk(K)  # (batch_size, seq_len_k, dm)
        V = self.Wv(V)  # (batch_size, seq_len_v, dm)

        # Reshape to (batch_size, h, seq_len, depth)
        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        # Compute the scaled dot product attention from Q, K, V
        scaled_attention, attention_weights = sdp_attention(Q, K, V, mask)
        # scaled_attention: shape (batch_size, h, seq_len_q, depth)
        # attention_weights: shape (batch_size, h, seq_len_q, seq_len_k)

        # Reshape to (batch_size, seq_len_q, h, depth)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])

        # Concatenate by reshaping (..., h, depth) to (..., dm)
        concatenated_attention = tf.reshape(scaled_attention,
                                            (batch_size, -1, self.dm))

        # Pass the concatenated_attention (output) to the 'linear' Dense layer
        output = self.linear(concatenated_attention)

        # Note: at each location in (each of) the sequence(s),
        # the MultiHeadAttention block runs all 'h' attention heads
        # across all other locations in the sequence, returning
        # a new vector (same length) at each location.

        return output, attention_weights
