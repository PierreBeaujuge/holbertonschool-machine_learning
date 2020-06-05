#!/usr/bin/env python3
"""
Multiple Kernels
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """function that performs a convolution on images using multiple kernels"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
    elif padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # output size depends on filter size and must be equal to image size
        # imposing constraints on padding for a given set of strides
        ph = int(np.ceil(((sh * h) - sh + kh - h) / 2))
        pw = int(np.ceil(((sw * w) - sw + kw - w) / 2))

    if isinstance(padding, tuple) or padding == 'same':
        # pad images before convolution, padding always symmetric here
        images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    output = np.zeros(shape=(m,
                             int((h - kh + 2 * ph) / sh + 1),
                             int((w - kw + 2 * pw) / sw + 1),
                             nc))

    for k in range(nc):
        for i in range(int((h - kh + 2 * ph) / sh + 1)):
            for j in range(int((w - kw + 2 * pw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    k
                ] = np.sum(
                    images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * kernels[:, :, :, k],
                    axis=(1, 2, 3)
                )
    return output
