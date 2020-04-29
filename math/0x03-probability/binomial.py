#!/usr/bin/env python3
"""
Binomial distribution
"""


class Binomial:
    """define class"""

    e = 2.7182818285
    pi = 3.1415926536

    def __init__(self, data=None, n=1, p=0.5):
        """class constructor"""
        if data is None and isinstance(
                n, (float, int)) and isinstance(p, (float, int)):
            if n <= 0:
                raise ValueError('n must be a positive value')
            if p < 0 or p > 1:
                raise ValueError('p must be greater than 0 and less than 1')
            self.n = int(n)
            self.p = float(p)
        elif data is not None:
            if not isinstance(data, list):
                raise TypeError('data must be a list')
            if len(data) < 2:
                raise ValueError('data must contain multiple values')
            mean = float(sum(data) / len(data))
            stddev = float((sum(
                [(data[i] - mean) ** 2 for i in range(len(data))]
            ) / len(data)) ** 0.5)
            # mean = self.n * self.p # as per binomial dist laws
            # stddev = self.n * self.p * (1 - self.p) # as per binom. dist laws
            # then:
            # self.p = float(mean / ( stddev ** 2 / (self.p (1 - self.p))))
            # self.p = float(mean * ((self.p (1 - self.p)) / stddev ** 2))
            # 1 = float(mean * ((1 - self.p) / stddev ** 2))
            # stddev ** 2 / mean = 1 - self.p
            self.p = 1 - ((stddev ** 2) / mean)
            self.n = int(round(mean / self.p))
            self.p = float(mean / self.n)

    def pmf(self, k):
        """
        function that calculates the probability mass function
        for k successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k is None or k < 0 or k > self.n:
            return 0
        return (self.fact(self.n) / (self.fact(k) * self.fact(self.n - k))) * (
            (self.p ** k) * (1 - self.p) ** (self.n - k))

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
        if k is None or k < 0 or k > self.n:
            return 0
        return sum([(self.fact(self.n) /
                     (self.fact(i) * self.fact(self.n - i)))
                    * ((self.p ** i) * (1 - self.p) ** (self.n - i))
                    for i in range(0, k + 1)])
