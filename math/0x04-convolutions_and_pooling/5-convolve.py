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
    ph = padding[0]
    pw = padding[1]
    sh = stride[0]
    sw = stride[1]

    if isinstance(padding, tuple):
        output = np.zeros(shape=(m,
                                 int((h - kh + 2 * ph) / sh + 1),
                                 int((w - kw + 2 * pw) / sw + 1),
                                 nc))
        for k in range(nc):
            for i in range(int((h - kh) / sh + 1)):
                for j in range(int((w - kw) / sw + 1)):
                    output[
                        image_num,
                        i + ph,
                        j + pw,
                        k
                    ] = np.sum(
                        images[
                            image_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            :
                        ] * kernels[:, :, :, k],
                        axis=(1, 2, 3)
                    )
        return output
    elif padding == 'valid':
        output = np.zeros(shape=(m,
                                 int((h - kh) / sh + 1),
                                 int((w - kw) / sw + 1),
                                 nc))
        for k in range(nc):
            for i in range(int((h - kh) / sh + 1)):
                for j in range(int((w - kw) / sw + 1)):
                    output[
                        image_num,
                        i,
                        j,
                        k
                    ] = np.sum(
                        images[
                            image_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            :
                        ] * kernels[:, :, :, k],
                        axis=(1, 2, 3)
                    )
        return output
    elif padding == 'same':
        output = np.zeros(shape=(m,
                                 int(h / sh),
                                 int(w / sw),
                                 nc))
        for k in range(nc):
            for i in range(int((h - kh) / sh + 1)):
                for j in range(int((w - kw) / sw + 1)):
                    output[
                        image_num,
                        i + int((kh - 1) / 2),
                        j + int((kw - 1) / 2),
                        k
                    ] = np.sum(
                        images[
                            image_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            :
                        ] * kernels[:, :, :, k],
                        axis=(1, 2, 3)
                    )
        return output
