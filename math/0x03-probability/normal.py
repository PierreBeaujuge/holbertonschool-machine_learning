#!/usr/bin/env python3
"""
Normal distribution
"""


class Normal:
    """define class"""

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """class constructor"""
        if data is None and isinstance(
                mean, (float, int)) and isinstance(stddev, (float, int)):
            if stddev <= 0:
                raise ValueError('stddev must be a positive value')
            self.mean = float(mean)
            self.stddev = float(stddev)
        elif data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            self.mean = float(sum(data) / len(data))
            self.stddev = float((sum(
                [(data[i] - self.mean) ** 2 for i in range(len(data))]
            ) / len(data)) ** 0.5)

    def z_score(self, x):
        """function that calculates the z-score of a given x-value"""
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """function that calculates the x-value of a given z-score"""
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        function that calculates the probability density function
        for a given x-value
        """
        return (1 / (self.stddev * (2 * Normal.pi) ** 0.5) * (
            Normal.e ** ((-0.5) * ((x - self.mean) / self.stddev) ** 2)
        ))

    def cdf(self, x):
        """
        function that calculates the cumulative distribution function
        for a given x-value
        """
        return 0.5 * (1 + self.erf(
            ((x - self.mean) / (self.stddev * (2 ** 0.5)))
        ))

    def erf(self, x):
        """function that calculates the erf of x"""
        return (2 / Normal.pi ** 0.5) * (x -
                                         (x ** 3) / 3 +
                                         (x ** 5) / 10 -
                                         (x ** 7) / 42 +
                                         (x ** 9) / 216)
