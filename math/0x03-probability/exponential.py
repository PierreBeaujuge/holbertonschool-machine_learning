#!/usr/bin/env python3
"""
Exponential distribution
"""


class Exponential:
    """define class"""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """class constructor"""
        if data is None and isinstance(lambtha, (float, int)):
            if lambtha <= 0:
                raise ValueError('lambtha must be a positive value')
            self.lambtha = float(lambtha)
        elif data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """
        function that calculates the probability density function
        for a given time period x
        """
        if x is None or x < 0:
            return 0
        return (self.lambtha * (
            Exponential.e ** ((-1 * self.lambtha) * x)
        ))

    def cdf(self, x):
        """
        function that calculates the cumulative distribution function
        for a given time period x
        """
        if x is None or x < 0:
            return 0
        return (1 - (
            Exponential.e ** ((-1 * self.lambtha) * x)
        ))
