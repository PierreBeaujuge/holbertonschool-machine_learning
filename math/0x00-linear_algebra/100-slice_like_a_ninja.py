#!/usr/bin/env python3
# import numpy as np


def np_slice(matrix, axes={}):
    """function that slices a matrix along a specific axis"""
    if not axes:
        return matrix
    buffer_list = []
    for i in range(len(matrix.shape)):
        flag = 0
        for k, v in axes.items():
            if k == i:
                buffer_list.append(slice(*v))
                flag = 1
                break
        if flag == 0:
            buffer_list.append(slice(None, None, None))
    # buffer_list.append(slice(None, None, None))
    # return buffer_list
    # return list(map(lambda x, y: y if list(axes.keys())[x] ==
    # , list(axes.keys()), list(axes.values()) ))
    # return matrix[tuple(map(lambda x: x, buffer_list))]
    return matrix[tuple(buffer_list)]
    # return tuple(buffer_list)

    # elif list(axes.keys())[0] == 0:
    #     return matrix[list(axes.values())[0][0]:list(axes.values())[0][1]]
    # elif list(axes.keys())[0] == 1:
    #     # return matrix[:, list(axes.values())[0][0]:
    # list(axes.values())[0][1]]
    #     return matrix.slice().slice(list(axes.values())[0][0],
    # list(axes.values())[0][1])
