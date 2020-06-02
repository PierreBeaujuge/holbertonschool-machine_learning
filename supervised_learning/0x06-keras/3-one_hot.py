#!/usr/bin/env python3
"""
One Hot
"""
import tensorflow as tf


def one_hot(labels, classes=None):
    """function that converts a label vector into a one-hot matrix"""
    return tf.keras.utils.to_categorical(labels, num_classes=classes)
