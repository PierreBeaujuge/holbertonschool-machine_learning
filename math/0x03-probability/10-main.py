#!/usr/bin/env python3

import numpy as np
Binomial = __import__('binomial').Binomial

np.random.seed(0)
data = np.random.binomial(50, 0.6, 100).tolist()
b1 = Binomial(data)
print('n:', b1.n, "p:", b1.p)

b2 = Binomial(n=50, p=0.6)
print('n:', b2.n, "p:", b2.p)

b3 = Binomial(n=50, p=0)
print('n:', b3.n, "p:", b3.p)

b4 = Binomial(n=50, p=1)
print('n:', b4.n, "p:", b4.p)
