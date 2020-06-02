#!/usr/bin/env python3
"""
One Hot
"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """function that converts a label vector into a one-hot matrix"""
    return K.utils.to_categorical(labels, num_classes=classes)
