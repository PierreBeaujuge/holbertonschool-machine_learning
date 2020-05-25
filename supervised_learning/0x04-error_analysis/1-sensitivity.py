#!/usr/bin/env python3
"""
Sensitivity
"""
import numpy as np


def sensitivity(confusion):
    """function that calculates the sensitivity for each class"""
    classes, classes = confusion.shape
    sensitivity = np.zeros(shape=(classes,))
    for i in range(classes):
        sensitivity[i] = confusion[i][i] / np.sum(confusion, axis=1)[i]
    return sensitivity
