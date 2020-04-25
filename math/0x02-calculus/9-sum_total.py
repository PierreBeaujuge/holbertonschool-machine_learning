#!/usr/bin/env python3
"""
sum functions
"""


def summation_i_squared(n):
    """function that calculates summation_i_squared"""
    if isinstance(n, int) and n > 0:
        # return sum([i**2 for i in range(1, n + 1)])
        return int((n / 6) * (n + 1) * (2 * n + 1))
    return None
