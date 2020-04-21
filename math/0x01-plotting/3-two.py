#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 21000, 1000)
r = np.log(0.5)
t1 = 5730
t2 = 1600
y1 = np.exp((r / t1) * x)
y2 = np.exp((r / t2) * x)

fig, ax = plt.subplots()
y1 = ax.plot(x, y1, 'r--', label='C-14')
y2 = ax.plot(x, y2, 'g', label='Ra-226')
plt.autoscale(axis='x', tight=True)
plt.autoscale(axis='y', tight=True)
plt.xlabel('Time (years)')
plt.ylabel('Fraction Remaining')
plt.title('Exponential Decay of Radioactive Elements')
ax.legend()
plt.show()
