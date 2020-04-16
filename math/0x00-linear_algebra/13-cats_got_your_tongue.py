#!/usr/bin/env python3
"""define new function"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """function that concatenates two matrices along a specific axis"""
    return np.concatenate((mat1, mat2), axis)
