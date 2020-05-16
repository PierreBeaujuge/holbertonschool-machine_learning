#!/usr/bin/env python3
"""
Loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """function that calculates the softmax cross-entropy loss of pred"""
    return tf.losses.softmax_cross_entropy(onehot_labels=y, logits=y_pred)
