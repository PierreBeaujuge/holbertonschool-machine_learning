#!/usr/bin/env python3
"""
Convolution with Padding
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """function that performs a convolution with custom padding"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    ph = padding[0]
    pw = padding[1]
    output = np.zeros(shape=(m,
                             h - kh + 1 + 2 * padding[0],
                             w - kw + 1 + 2 * padding[1]))

    for i in range(h - kh + 1):
        for j in range(w - kw + 1):
            output[
                image_num,
                i + ph,
                j + pw
            ] = np.sum(
                images[
                    image_num,
                    i: i + kh,
                    j: j + kw
                ] * kernel,
                axis=(1, 2)
            )
    return output
