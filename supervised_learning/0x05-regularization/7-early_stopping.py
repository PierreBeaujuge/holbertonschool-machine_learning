#!/usr/bin/env python3
"""
Early Stopping
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """function that determines if you should stop gradient descent early"""
    early_stopping = False
    if opt_cost - cost <= threshold:
        count += 1
    else:
        count = 0
    if count == patience:
        early_stopping = True
        return (early_stopping, count)
    return (early_stopping, count)
