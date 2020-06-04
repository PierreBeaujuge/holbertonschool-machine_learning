#!/usr/bin/env python3
"""
Valid Convolution
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """function that performs a valid convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    output = np.zeros(shape=(m,
                             h - kh + 1,
                             w - kw + 1))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[image_num, i, j] = np.sum(
                images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output
