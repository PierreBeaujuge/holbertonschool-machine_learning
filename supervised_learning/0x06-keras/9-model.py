#!/usr/bin/env python3
"""
Save and Load Model
"""
import tensorflow as tf


def save_model(network, filename):
    """saves an entire model"""
    network.save(
        filepath=filename,
        overwrite=True,
        include_optimizer=True
    )
    return None


def load_model(filename):
    """loads an entire model"""
    network = tf.keras.models.load_model(
        filepath=filename,
        custom_objects=None,
        compile=True
    )
    return network
