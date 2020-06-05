#!/usr/bin/env python3
"""
Strided Convolution
"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """function that performs a convolution with custom padding"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]
        # pad images before convolution, padding always symmetric here
        padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                               mode='constant')
        # output size depends on filter size and padding
        output = np.zeros(shape=(m,
                                 int((h - kh + 2 * ph) / sh + 1),
                                 int((w - kw + 2 * pw) / sw + 1)))

        for i in range(int((h - kh + 2 * ph) / sh + 1)):
            for j in range(int((w - kw + 2 * pw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j
                ] = np.sum(
                    padded_images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * kernel,
                    axis=(1, 2)
                )
        return output
    elif padding == 'valid':
        output = np.zeros(shape=(m,
                                 int((h - kh) / sh + 1),
                                 int((w - kw) / sw + 1)))

        for i in range(int((h - kh) / sh + 1)):
            for j in range(int((w - kw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j
                ] = np.sum(
                    images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * kernel,
                    axis=(1, 2)
                )
        return output
    elif padding == 'same':
        output = np.zeros(shape=(m,
                                 int(h / sh),
                                 int(w / sw)))
        # pad images before convolution
        # distinguish between even and odd filter sizes
        if kh % 2 == 0:
            ph = int(kh/2)
        else:
            ph = int((kh - 1)/2)
        if kw % 2 == 0:
            pw = int(kw/2)
        else:
            pw = int((kw - 1)/2)
        # pad images accordingly, padding always symmetric here
        padded_images = np.pad(images, pad_width=((0, 0), (ph, ph), (pw, pw)),
                               mode='constant')

        for i in range(int(h / sh)):
            for j in range(int(w / sw)):
                output[
                    image_num,
                    i,
                    j
                ] = np.sum(
                    padded_images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * kernel,
                    axis=(1, 2)
                )
        return output
