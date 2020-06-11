#!/usr/bin/env python3
"""
Forward Convolution
"""
import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """function that performs a forward propagation over a convolution layer"""
    m = A_prev.shape[0]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    c_prev = W.shape[2]
    c_new = W.shape[3]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]

    if padding == 'valid':
        ph = 0
        pw = 0
    elif padding == 'same':
        # output size depends on filter size and must be equal to image size
        # imposing constraints on padding for a given set of strides
        ph = int(np.ceil(((sh * h_prev) - sh + kh - h_prev) / 2))
        pw = int(np.ceil(((sw * w_prev) - sw + kw - w_prev) / 2))

    if padding == 'same':
        # pad A_prev before convolution, padding always symmetric here
        A_prev = np.pad(A_prev, pad_width=((0, 0), (ph, ph), (pw, pw), (0, 0)),
                        mode='constant')

    output = np.zeros(shape=(m,
                             int((h_prev - kh + 2 * ph) / sh + 1),
                             int((w_prev - kw + 2 * pw) / sw + 1),
                             c_new))

    for k in range(c_new):
        for i in range(int((h_prev - kh + 2 * ph) / sh + 1)):
            for j in range(int((w_prev - kw + 2 * pw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    k
                ] = np.sum(
                    A_prev[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw
                    ] * W[:, :, :, k],
                    axis=(1, 2, 3)
                ) + b[0, 0, 0, k]
                # output[
                #     image_num,
                #     i,
                #     j,
                #     k
                # ] = activation(output[
                #     image_num,
                #     i,
                #     j,
                #     k])
    output = activation(output)
    return output
