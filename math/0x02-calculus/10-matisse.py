#!/usr/bin/env python3
"""
function poly_derivative
"""
import numpy as np


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    if poly and all(isinstance(x, (int, float)) for x in poly):
        result = [poly[i] * i for i in range(1, len(poly))]
        if not len(result):
            return [0]
        return result
    return None
