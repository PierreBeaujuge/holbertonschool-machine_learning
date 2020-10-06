#!/usr/bin/env python3
"""
4-create_masks.py
"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """function that creates all masks for training/validation"""

    # Mask all the pad tokens: ensures that the model does not
    # treat padding as the input. The mask indicates where pad value 0
    # is present: it outputs a 1 at those locations, and a 0 otherwise.

    # Replace 0 pad tokens by 1s
    inputs = tf.cast(tf.math.equal(inputs, 0), tf.float32)

    # add extra dimensions to add the padding to the attention logits.

    # Encoder padding mask
    # Used in the attention block in the encoder
    encoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    # (batch_size, 1, 1, seq_len_in)

    # Decoder padding mask
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    decoder_mask = inputs[:, tf.newaxis, tf.newaxis, :]
    # (batch_size, 1, 1, seq_len_in)

    # Look-ahead mask:
    # used to mask the future tokens in a sequence. In other words,
    # the mask indicates which entries should not be used.
    # Used in the 1st attention block in the decoder
    # --> pad and mask future tokens in the input received by
    # the decoder
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((target.shape[0], 1, target.shape[1], target.shape[1])), -1, 0)
    # (seq_len_in, 1, seq_len_out, seq_len_out)
    look_ahead_mask = tf.maximum(decoder_mask, look_ahead_mask)

    return encoder_mask, look_ahead_mask, decoder_mask
