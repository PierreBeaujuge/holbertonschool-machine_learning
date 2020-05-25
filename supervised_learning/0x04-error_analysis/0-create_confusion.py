#!/usr/bin/env python3
"""
Create Confusion
"""
import numpy as np


def create_confusion_matrix(labels, logits):
    """function that creates a confusion matrix"""
    m, classes = labels.shape
    v1 = np.argmax(labels, axis=1)
    v2 = np.argmax(logits, axis=1)
    confusion = np.zeros(shape=(classes, classes))
    for i in range(classes):
        for j in range(classes):
            for k in range(m):
                if i == v1[k] and j == v2[k]:
                    confusion[i][j] += 1
    return confusion
