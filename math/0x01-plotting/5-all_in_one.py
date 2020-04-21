#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

plt.suptitle('All in One')

y1 = np.arange(0, 11) ** 3

ax1 = plt.subplot(321)
plt.plot(y1, c='r')
plt.autoscale(axis='x', tight=True)

mean = [69, 0]
cov = [[15, 8], [8, 15]]
np.random.seed(5)
x2, y2 = np.random.multivariate_normal(mean, cov, 2000).T
y2 += 180

ax2 = plt.subplot(322)
ax2.scatter(x2, y2, c='magenta', s=8)
ax2.set_xlabel('Height (in)', fontsize='x-small')
ax2.set_ylabel('Weight (lbs)', fontsize='x-small')
ax2.set_title('Men\'s Height vs Weight', fontsize='x-small')

x3 = np.arange(0, 28651, 5730)
r3 = np.log(0.5)
t3 = 5730
y3 = np.exp((r3 / t3) * x3)

ax3 = plt.subplot(323)
ax3.plot(x3, y3)
ax3.set_yscale('log')
ax3.set_xlim(0, 28650)
ax3.set_xlabel('Time (years)', fontsize='x-small')
ax3.set_ylabel('Fraction Remaining', fontsize='x-small')
ax3.set_title('Exponential Decay of C-14', fontsize='x-small')

x4 = np.arange(0, 21000, 1000)
r4 = np.log(0.5)
t14 = 5730
t24 = 1600
y14 = np.exp((r4 / t14) * x4)
y24 = np.exp((r4 / t24) * x4)

ax4 = plt.subplot(324)
ax4.plot(x4, y14, 'r--', x4, y24, 'g')
ax4.set_xlim(0, 20000)
ax4.set_ylim(0, 1)
ax4.set_xlabel('Time (years)', fontsize='x-small')
ax4.set_ylabel('Fraction Remaining', fontsize='x-small')
ax4.set_title('Exponential Decay of Radioactive Elements', fontsize='x-small')
ax4.legend(('C-14', 'Ra-226'), fontsize='x-small')

np.random.seed(5)
student_grades = np.random.normal(68, 15, 50)

ax5 = plt.subplot(313)
ax5.hist(student_grades, edgecolor='black', bins=range(0, 110, 10))
ax5.set_xlim(0, 100)
ax5.set_xticks(np.arange(0, 110, step=10))
ax5.set_ylim(0, 30)
ax5.set_xlabel('Grades', fontsize='x-small')
ax5.set_ylabel('Number of Students', fontsize='x-small')
ax5.set_title('Project A', fontsize='x-small')

plt.tight_layout()
plt.show()
