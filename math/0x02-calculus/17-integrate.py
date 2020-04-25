#!/usr/bin/env python3
"""
integration functions
"""


def poly_integral(poly, C=0):
    """function that calculates the integral of a polynomial"""
    if poly and isinstance(poly, list) and (
            isinstance(C, int) or isinstance(C, float)) and all(
            isinstance(x, (int, float)) for x in poly):
        if poly == [0]:
            return [C]
        arr = [float(C)] + [poly[i] / (i + 1) for i in range(0, len(poly))]
        return [float(x) if not x.is_integer() else int(x) for x in arr]
    return None
