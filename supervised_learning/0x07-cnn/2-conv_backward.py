#!/usr/bin/env python3
"""
Backpropagation over Convolution Layer
"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """function that performs a backpropagation over a convolutional layer"""
    m = dZ.shape[0]
    h_new = dZ.shape[1]
    w_new = dZ.shape[2]
    c_new = dZ.shape[3]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    c_prev = A_prev.shape[3]
    kh = W.shape[0]
    kw = W.shape[1]
    # image_num = np.arange(m)
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

    dA_prev = np.zeros(shape=A_prev.shape)
    dW = np.zeros(shape=W.shape)
    db = np.zeros(shape=b.shape)

    for img_num in range(m):
        for k in range(c_new):
            for i in range(h_new):
                for j in range(w_new):
                    dA_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] += dZ[
                        img_num,
                        i,
                        j,
                        k
                    ] * W[:, :, :, k]
                    dW[:, :, :, k] += A_prev[
                        img_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ] * dZ[
                        img_num,
                        i,
                        j,
                        k
                    ]
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    if padding == 'same':
        dA_prev = dA_prev[
            :,
            ph: dA_prev.shape[1] - ph,
            pw: dA_prev.shape[2] - pw,
            :
        ]
    return dA_prev, dW, db
