#!/usr/bin/env python3
"""
Momentum Upgraded
"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """function that implements momentum gradient descent in tensorflow"""
    return tf.train.MomentumOptimizer(
        learning_rate=alpha, momentum=beta1, use_locking=False,
        name='Momentum', use_nesterov=False
    ).minimize(loss)
