#!/usr/bin/env python3
"""
Backpropagation over Pooling Layer
"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """function that performs a backpropagation over a convolutional layer"""
    m = dA.shape[0]
    h_new = dA.shape[1]
    w_new = dA.shape[2]
    c = dA.shape[3]
    h_prev = A_prev.shape[1]
    w_prev = A_prev.shape[2]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    # image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]
    func = {'max': np.max, 'avg': np.mean}

    dA_prev = np.zeros(shape=A_prev.shape)

    if mode in ['max', 'avg']:
        for img_num in range(m):
            for k in range(c):
                for i in range(h_new):
                    for j in range(w_new):
                        window = A_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ]
                        if mode == 'max':
                            # maxpool returns the max
                            # derivative of maxpool relative to the max is 1
                            # derivative relative to any other element is 0
                            # backpropagate 1 to the unit corresponding to max
                            # backpropagate 0 for the other units
                            # given these comments, define a mask of 1 and 0s
                            mask = np.where(window == np.max(window), 1, 0)
                            # print(mask)
                        elif mode == 'avg':
                            # define a mask weighted by the number of
                            # elements in the pooling layer (kh * kw)
                            mask = np.ones(shape=window.shape)
                            mask /= (kh * kw)
                            # print(mask)
                        dA_prev[
                            img_num,
                            i * sh: i * sh + kh,
                            j * sw: j * sw + kw,
                            k
                        ] += mask * dA[
                            img_num,
                            i,
                            j,
                            k
                        ]
    return dA_prev
