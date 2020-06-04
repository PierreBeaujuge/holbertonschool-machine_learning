#!/usr/bin/env python3
"""
Pooling
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """function that performs pooling on images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]
    kh = kernel_shape[0]
    kw = kernel_shape[1]
    image_num = np.arange(m)
    sh = stride[0]
    sw = stride[1]
    func = {'max': np.max, 'avg': np.mean}

    output = np.zeros(shape=(m,
                             int((h - kh) / sh + 1),
                             int((w - kw) / sw + 1),
                             c))
    if mode in ['max', 'avg']:
        for i in range(int((h - kh) / sh + 1)):
            for j in range(int((w - kw) / sw + 1)):
                output[
                    image_num,
                    i,
                    j,
                    :
                ] = func[mode](
                    images[
                        image_num,
                        i * sh: i * sh + kh,
                        j * sw: j * sw + kw,
                        :
                    ], axis=(1, 2)
                )
    return output
