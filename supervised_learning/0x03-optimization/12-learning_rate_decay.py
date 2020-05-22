#!/usr/bin/env python3
"""
Learning Rate Decay Upgraded
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """function that updates the learning rate using inverse time decay"""
    return tf.train.inverse_time_decay(
        learning_rate=alpha, global_step=global_step, decay_steps=decay_step,
        decay_rate=decay_rate, staircase=True, name=None
    )
