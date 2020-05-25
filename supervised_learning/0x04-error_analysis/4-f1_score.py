#!/usr/bin/env python3
"""
F1 score
"""
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """function that calculates the F1 score of a confusion matrix"""
    s = sensitivity(confusion)
    p = precision(confusion)
    f1 = 2 * (p * s) / (p + s)
    return f1
