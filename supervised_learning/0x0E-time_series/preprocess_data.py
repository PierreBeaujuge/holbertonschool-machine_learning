#!/usr/bin/env python3
"""
Time Series Forecasting

The following script can be run on Google Colab
with the following libraries:

!python3 --version
Python 3.6.9
print(tf.__version__)
2.3.0
print(np.__version__)
1.18.5
print(matplotlib.__version__)
3.2.2
print(pd.__version__)
1.0.5
print(keras.__version__)
2.4.3

Alternative library versioning may incur errors
due to compatibility issues between deprecated
libraries on Google Colab
"""
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=np.inf)
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import math


def preprocess_data(file_path):
    """function that preprocesses the data from the dataset"""

    df = pd.read_csv(file_path)
    # Remove all NaN-containing entries
    df = df.dropna()
    # Reformat the timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'],
                                     infer_datetime_format=True,
                                     unit='s')
    # Use the timestamp as index column
    df = df.set_index('Timestamp')
    # Narrow the scope of dataframe columns to work with
    df = df.drop(['Low', 'High', 'Volume_(BTC)', 'Weighted_Price'], axis=1)
    # Reorder columns
    df = df.reindex(columns=['Open', 'Close', 'Volume_(Currency)'])
    # Work on a 1hour window
    df['Open'] = df['Open'].resample('1H').first()
    df['Close'] = df['Close'].resample('1H').last()
    df['Volume_(Currency)'] = df['Volume_(Currency)'].resample('1H').sum()
    # Remove all NaN-containing entries
    df = df.dropna()
    # Remove the first half of the dataframe (given data sparsity)
    df = df.iloc[-int((df.shape[0]/2)):]

    print(df.head(10))
    print('=================')

    # Create the dataset (np.ndarray of "df.shape")
    dataset = df.values
    print(dataset[:10])
    print(dataset.shape)
    print('=======================')

    # Standardize the dataset
    mean = np.mean(dataset, axis=0)
    stddev = np.std(dataset, axis=0)
    dataset = (dataset - mean) / stddev
    print(dataset[10:])
    print(dataset.shape)
    print('=======================')

    def split_sequence(sequence, n_steps):
        """function that splits a dataset sequence into input data and
        labels"""

        X, Y = [], []
        for i in range(sequence.shape[0]):
            if (i + n_steps) >= sequence.shape[0]:
                break
            # Divide sequence between data (input) and labels (output)
            seq_X, seq_Y = sequence[i: i + n_steps], sequence[i + n_steps, -2]
            X.append(seq_X)
            Y.append(seq_Y)

        return np.array(X), np.array(Y)

    # Create training and validation datasets
    dataset_size = dataset.shape[0]
    x_train, y_train = split_sequence(
        dataset[0: math.ceil(0.7 * dataset_size)], 24)
    x_val, y_val = split_sequence(
        dataset[math.floor(0.7 * dataset_size):], 24)

    return dataset, df, x_train, y_train, x_val, y_val
