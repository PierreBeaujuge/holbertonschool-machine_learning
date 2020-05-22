#!/usr/bin/env python3
"""
Momentum
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """function that implements  gradient descent with momentum optimization"""
    v = beta1 * v + (1 - beta1) * grad
    var -= alpha * v
    return var, v
