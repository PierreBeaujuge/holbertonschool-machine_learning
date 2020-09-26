#!/usr/bin/env python3
"""
5-sdp_attention.py
"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """function that computes the scaled dot product (sdp) attention"""

    # Q: query shape == (..., seq_len_q, depth)
    # K: key shape == (..., seq_len_k, depth)
    # V: value shape == (..., seq_len_v, depth_v)
    # mask: Float tensor with shape broadcastable
    # to (..., seq_len_q, seq_len_k)

    # Compute the dot product of Q and K (attention_logits)
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # (..., seq_len_q, seq_len_k)

    # Scale matmul_qk (attention_logits)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # Add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # Softmax over the last axis (seq_len_k) so that the scores add up to 1
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # attention_weights: shape (..., seq_len_q, seq_len_k)

    # Note: As the softmax normalization is done on K, its values decide
    # the amount of significance given to Q.
    # The output represents the multiplication of the attention weights and
    # the V (value) vector. This ensures that the words you want to focus on
    # are kept as-is and the irrelevant words are flushed out.

    # Note: seq_len_k == seq_len_v
    output = tf.matmul(attention_weights, V)  # (..., seq_len_q, depth_v)

    return output, attention_weights
