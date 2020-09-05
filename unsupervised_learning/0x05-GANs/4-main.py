#!/usr/bin/env python3
import tensorflow as tf
sample_Z = __import__('4-sample_Z').sample_Z


if __name__ == "__main__":

    Z = sample_Z(5, 100)

    print(Z)
