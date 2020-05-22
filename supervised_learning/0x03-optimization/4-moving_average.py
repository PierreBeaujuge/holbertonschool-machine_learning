#!/usr/bin/env python3
"""
Moving Average
"""


def moving_average(data, beta):
    """function that calculates the weighted moving average of a data set"""
    mov_avgs = []
    mov_avg = 0
    for i in range(len(data)):
        # bias correction: 1 - beta ** k; with k: iteration number
        mov_avg = beta * mov_avg + (1 - beta) * data[i]
        mov_avgs += [mov_avg / (1 - beta ** (i + 1))]
    return mov_avgs
