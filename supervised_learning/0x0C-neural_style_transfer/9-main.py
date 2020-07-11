#!/usr/bin/env python3

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

NST = __import__('9-neural_style').NST


if __name__ == '__main__':
    style_image = mpimg.imread("starry_night.jpg")
    content_image = mpimg.imread("golden_gate.jpg")

    np.random.seed(0)
    nst = NST(style_image, content_image)
    image, cost = nst.generate_image(iterations=2000, step=100, lr=0.002)
    print("Best cost:", cost)
    plt.imshow(image)
    plt.show()
