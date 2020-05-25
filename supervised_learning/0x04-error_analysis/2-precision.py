#!/usr/bin/env python3
"""
Precision
"""
import numpy as np


def precision(confusion):
    """function that calculates the precision for each class"""
    classes, classes = confusion.shape
    precision = np.zeros(shape=(classes,))
    for i in range(classes):
        precision[i] = confusion[i][i] / np.sum(confusion, axis=0)[i]
    return precision
