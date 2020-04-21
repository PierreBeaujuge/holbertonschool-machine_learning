#!/usr/bin/env python3
"""
function poly_derivative
"""
import numpy as np


def poly_derivative(poly):
    """function that calculates the derivative of a polynomial"""
    return [poly[i] * i for i in range(1, len(poly))]
