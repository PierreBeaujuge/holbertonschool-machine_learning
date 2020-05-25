#!/usr/bin/env python3
"""
Specificity
"""
import numpy as np


def specificity(confusion):
    """function that calculates the specificity for each class"""
    classes, classes = confusion.shape
    specificity = np.zeros(shape=(classes,))
    for i in range(classes):
        specificity[i] = (
            np.sum(confusion) - np.sum(confusion, axis=1)[i]
            - np.sum(confusion, axis=0)[i] + confusion[i][i]
        ) / (np.sum(confusion) - np.sum(confusion, axis=1)[i])
    return specificity
