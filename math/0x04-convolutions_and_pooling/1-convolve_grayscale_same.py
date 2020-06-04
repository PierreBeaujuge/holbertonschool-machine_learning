#!/usr/bin/env python3
"""
Same Convolution
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """function that performs a same convolution on grayscale images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    output = np.zeros(shape=(m, h, w))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[
                image_num,
                i + int((kh - 1) / 2),
                j + int((kw - 1) / 2)
            ] = np.sum(
                images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output
