#!/usr/bin/env python3
"""
Adam
"""
import numpy as np


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """function that implements the Adam optimization algorithm"""
    v = beta1 * v + (1 - beta1) * grad
    # introduce bias correction
    v_corr = v / (1 - (beta1 ** t))
    s = beta2 * s + (1 - beta2) * (grad ** 2)
    # introduce bias correction
    s_corr = s / (1 - (beta2 ** t))
    var -= alpha * (v_corr / (np.sqrt(s_corr) + epsilon))
    return var, v, s
