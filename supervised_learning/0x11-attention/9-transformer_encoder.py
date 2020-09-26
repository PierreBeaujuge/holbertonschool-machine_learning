#!/usr/bin/env python3
"""
9-transformer_encoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
EncoderBlock = __import__('7-transformer_encoder_block').EncoderBlock


class Encoder(tf.keras.layers.Layer):
    """class that instantiates an Encoder"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor"""
        super(Encoder, self).__init__()

        # Number of blocks in the encoder
        self.N = N
        # Depth of the model
        self.dm = dm
        # Instantiate an embedding layer
        # note: input_vocab == vocab_size
        self.embedding = tf.keras.layers.Embedding(input_vocab, dm)
        # Instantiate a positional encoding layer
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # List with N encoder blocks
        self.blocks = [EncoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        # Instantiate a dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """function that builds an Encoder"""

        # seq_len = tf.shape(x)[1]
        seq_len = x.shape[1]

        # Compute the embeddings; shape (batch_size, input_seq_len, dm)
        embeddings = self.embedding(x)
        # Scale the embeddings
        embeddings *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        # Sum the positional encodings with the embeddings
        embeddings += self.positional_encoding[:seq_len, :]
        # Pass the embeddings on to the dropout layer
        output = self.dropout(embeddings, training=training)

        # Pass the output on to the N encoder blocks (one by one)
        for i in range(self.N):
            output = self.blocks[i](output, training, mask)

        # Note: shape of all output tensors (batch_size, input_seq_len, dm)

        return output
