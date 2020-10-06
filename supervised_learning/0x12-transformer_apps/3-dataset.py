#!/usr/bin/env python3
"""
0-dataset.py
"""
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """class that loads and preps a dataset for machine translation"""

    def __init__(self, batch_size, max_len):
        """constructor"""

        # Portugese-English translation dataset
        # from the TED Talks Open Translation Project
        # contains approximately 50000 training examples,
        # 1100 validation examples, and 2000 test examples

        # Download the dataset
        examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                                       with_info=True,
                                       as_supervised=True)
        # print(type(examples))
        # print(examples)
        # print(examples['train'])

        # Extract the train and validation datasets
        self.data_train = examples['train']
        self.data_valid = examples['validation']

        # Create sub-word tokenizers for the dataset
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train)

        # Update the data_train and data_validate attributes
        # by tokenizing the examples
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

        def filter_max_length(x, y, max_len=max_len):
            """helper function to .filter() method"""
            return tf.logical_and(tf.size(x) <= max_len,
                                  tf.size(y) <= max_len)

        self.data_train = self.data_train.filter(filter_max_length)
        # cache the dataset to memory to get a speedup while reading from it.
        self.data_train = self.data_train.cache()
        buffer_size = metadata.splits['train'].num_examples
        self.data_train = self.data_train.shuffle(buffer_size).padded_batch(
            batch_size, padded_shapes=([None], [None]))
        self.data_train = self.data_train.prefetch(
            tf.data.experimental.AUTOTUNE)

        self.data_valid = self.data_valid.filter(filter_max_length)
        self.data_valid = self.data_valid.padded_batch(
            batch_size, padded_shapes=([None], [None]))

    def tokenize_dataset(self, data):
        """function that creates sub-word tokenizers for a dataset"""

        tokenizer_en = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for pt, en in data),
            target_vocab_size=2**15)

        tokenizer_pt = tfds.features.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, en in data),
            target_vocab_size=2**15)

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """function that encodes a translation into tokens"""

        pt_tokens = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            pt.numpy()) + [self.tokenizer_pt.vocab_size + 1]
        en_tokens = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            en.numpy()) + [self.tokenizer_en.vocab_size + 1]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """function that wraps the 'encode' methods instance,
        so that it can be used with .map()"""

        # Need to use Dataset.map to apply the 'encode' method to each element
        # of the dataset. However, Dataset.map runs in graph mode.
        # Graph tensors do not have a value. In graph mode, one can only use
        # TensorFlow Ops and functions. So one cannot .map this 'encode'
        # function directly: It must be wrapped in a tf.py_function.
        # The tf.py_function will pass regular tensors (with a value and a
        # .numpy() method to access it), to the wrapped in the python function.

        result_pt, result_en = tf.py_function(self.encode,
                                              [pt, en],
                                              [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en
