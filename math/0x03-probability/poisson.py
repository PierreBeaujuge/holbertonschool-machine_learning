#!/usr/bin/env python3
"""
Poisson distribution
"""


class Poisson:
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
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        function that calculates the probability mass function
        for k successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0:
            return 0
        # not taken by pep8; no explanation
        # fact = lambda x: 1 if x == 0 else x * fact(x-1)
        return ((self.lambtha ** k) * (
            Poisson.e ** (-1 * self.lambtha)
        )) / self.fact(k)

    def fact(self, k):
        """function that returns the factorial of k"""
        if k in [0, 1]:
            return 1
        return k * self.fact(k - 1)

    def cdf(self, k):
        """
        function that calculates the cumulative distribution function
        for k successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0:
            return 0
        return Poisson.e ** (-1 * self.lambtha) * sum(
            [self.lambtha ** i / self.fact(i) for i in range(0, k + 1)])
