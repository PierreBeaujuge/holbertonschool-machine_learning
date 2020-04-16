#!/usr/bin/env python3
"""define new function"""


def add_arrays(arr1, arr2):
    """function that adds two arrays element-wise"""
    if len(arr1) == len(arr2):
        return list(map(lambda x, y: x + y, arr1, arr2))
        # list comprehension is a working alternative;
        # although both options return a new list:
        # return [arr1[i] + arr2[i] for i in range(len(arr1))]
    return None
