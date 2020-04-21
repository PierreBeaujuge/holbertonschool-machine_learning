#!/usr/bin/env python3
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

lib = np.load("pca.npz")
data = lib["data"]
labels = lib["labels"]

data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

fig = plt.figure()
ax = Axes3D(fig)
plt.plasma()
ax.scatter(pca_data.T[0], pca_data.T[1], pca_data.T[2], c=labels, s=40)
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
fig.suptitle("PCA of Iris Dataset")

plt.show()
