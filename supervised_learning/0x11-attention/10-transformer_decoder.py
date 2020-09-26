#!/usr/bin/env python3
"""
10-transformer_decoder.py
"""
import tensorflow as tf
positional_encoding = __import__('4-positional_encoding').positional_encoding
DecoderBlock = __import__('8-transformer_decoder_block').DecoderBlock


class Decoder(tf.keras.layers.Layer):
    """class that instantiates a Decoder"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """constructor"""
        super(Decoder, self).__init__()

        # Number of blocks in the decoder
        self.N = N
        # Depth of the model
        self.dm = dm
        # Instantiate an embedding layer
        # note: target_vocab == vocab_size
        self.embedding = tf.keras.layers.Embedding(target_vocab, dm)
        # Instantiate a positional encoding layer
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        # List with N decoder blocks
        self.blocks = [DecoderBlock(dm, h, hidden, drop_rate)
                       for _ in range(N)]
        # Instantiate a dropout layer
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """function that builds a Decoder"""

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
            output = self.blocks[i](output, encoder_output, training,
                                    look_ahead_mask, padding_mask)

        # Note: shape of all output tensors (batch_size, input_seq_len, dm)

        return output
