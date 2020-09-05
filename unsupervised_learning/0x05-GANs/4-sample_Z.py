#!/usr/bin/env python3
"""
4-sample_Z.py
"""
import numpy as np


def sample_Z(m, n):
    """function that creates input for the generator"""

    Z = np.random.uniform(-1, 1, size=(m, n))

    return Z
