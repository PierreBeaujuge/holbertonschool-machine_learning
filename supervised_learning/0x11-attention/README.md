# 0x11. Attention

## Learning Objectives

- What is the attention mechanism?
- How to apply attention to RNNs
- What is a transformer?
- How to create an encoder-decoder transformer model
- What is GPT?
- What is BERT?
- What is self-supervised learning?
- How to use BERT for specific NLP tasks
- What is SQuAD? GLUE?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15) and `tensorflow` (version 1.15)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise stated, you cannot import any module except import tensorflow as tf

## Update Tensorflow to 1.15

In order to complete the following tasks, you will need to update `tensorflow` to version 1.15, which will also update `numpy` to version 1.16

```
pip install --user tensorflow==1.15
```

## Tasks

### [0. RNN Encoder](./0-rnn_encoder.py)

Create a class `RNNEncoder` that inherits from `tensorflow.keras.layers.Layer` to encode for machine translation:
- Class constructor def `__init__(self, vocab, embedding, units, batch):`
  - `vocab` is an integer representing the size of the input vocabulary
  - `embedding` is an integer representing the dimensionality of the embedding vector
  - `units` is an integer representing the number of hidden units in the RNN cell
  - `batch` is an integer representing the batch size
  - Sets the following public instance attributes:
    - `batch` - the batch size
    - `units` - the number of hidden units in the RNN cell
    - `embedding` - a keras Embedding layer that converts words from the vocabulary into an embedding vector
    - `gru` - a `keras` GRU layer with `units` units
      - Should return both the full sequence of outputs as well as the last hidden state
      - Recurrent weights should be initialized with `glorot_uniform`
- Public instance method `def initialize_hidden_state(self):`
  - Initializes the hidden states for the RNN cell to a tensor of zeros
  - Returns: a tensor of shape `(batch, units)` containing the initialized hidden states
- Public instance method `def call(self, x, initial):`
  - `x` is a tensor of shape `(batch, input_seq_len)` containing the input to the encoder layer as word indices within the vocabulary
  - `initial` is a tensor of shape `(batch, units)` containing the initial hidden state
  - Returns: `outputs, hidden`
    - `outputs` is a tensor of shape `(batch, input_seq_len, units)` containing the outputs of the encoder
    - `hidden` is a tensor of shape `(batch, units)` containing the last hidden state of the encoder

```
$ ./0-main.py
32
256
<class 'tensorflow.python.keras.layers.embeddings.Embedding'>
<class 'tensorflow.python.keras.layers.recurrent.GRU'>
Tensor("zeros:0", shape=(32, 256), dtype=float32)
Tensor("rnn_encoder/gru/transpose_1:0", shape=(32, 10, 256), dtype=float32)
Tensor("rnn_encoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
```
Ignore the Warning messages in the output

### [1. Self Attention](./1-self_attention.py)

Create a class SelfAttention that inherits from tensorflow.keras.layers.Layer to calculate the attention for machine translation based on this paper:

- Class constructor def __init__(self, units):
  - units is an integer representing the number of hidden units in the alignment model
  - Sets the following public instance attributes:
    - W - a Dense layer with units units, to be applied to the previous decoder hidden state
    - U - a Dense layer with units units, to be applied to the encoder hidden states
    - V - a Dense layer with 1 units, to be applied to the tanh of the sum of the outputs of W and U
- Public instance method def call(self, s_prev, hidden_states):
  - s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
  - hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
  - Returns: context, weights
    - context is a tensor of shape (batch, units) that contains the context vector for the decoder
    - weights is a tensor of shape (batch, input_seq_len, 1) that contains the attention weights

```
$ ./1-main.py
<tensorflow.python.keras.layers.core.Dense object at 0x12309d3c8>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536b38>
<tensorflow.python.keras.layers.core.Dense object at 0xb28536e48>
Tensor("self_attention/Sum:0", shape=(32, 256), dtype=float64)
Tensor("self_attention/transpose_1:0", shape=(32, 10, 1), dtype=float64)
$
```
Ignore the Warning messages in the output

### [2. RNN Decoder](./2-rnn_decoder.py)

Create a class RNNDecoder that inherits from tensorflow.keras.layers.Layer to decode for machine translation:

- Class constructor def __init__(self, vocab, embedding, units, batch):
  - vocab is an integer representing the size of the output vocabulary
  - embedding is an integer representing the dimensionality of the embedding vector
  - units is an integer representing the number of hidden units in the RNN cell
  - batch is an integer representing the batch size
  - Sets the following public instance attributes:
    - embedding - a keras Embedding layer that converts words from the vocabulary into an embedding vector
    - gru - a keras GRU layer with units units
      - Should return both the full sequence of outputs as well as the last hidden state
      - Recurrent weights should be initialized with glorot_uniform
    - F - a Dense layer with vocab units
- Public instance method def call(self, x, s_prev, hidden_states):
  - x is a tensor of shape (batch, 1) containing the previous word in the target sequence as an index of the target vocabulary
  - s_prev is a tensor of shape (batch, units) containing the previous decoder hidden state
  - hidden_states is a tensor of shape (batch, input_seq_len, units)containing the outputs of the encoder
  - You should use SelfAttention = __import__('1-self_attention').SelfAttention
  - You should concatenate the context vector with x in that order
  - Returns: y, s
    - y is a tensor of shape (batch, vocab) containing the output word as a one hot vector in the target vocabulary
    - s is a tensor of shape (batch, units) containing the new decoder hidden state

```
$ ./2-main.py
<tensorflow.python.keras.layers.embeddings.Embedding object at 0x1321113c8>
<tensorflow.python.keras.layers.recurrent.GRU object at 0xb375aab00>
<tensorflow.python.keras.layers.core.Dense object at 0xb375d5128>
Tensor("rnn_decoder/dense/BiasAdd:0", shape=(32, 2048), dtype=float32)
Tensor("rnn_decoder/gru/while/Exit_2:0", shape=(32, 256), dtype=float32)
$
```
Ignore the Warning messages in the output

### [3. Positional Encoding](./4-positional_encoding.py)

Write the function def positional_encoding(max_seq_len, dm): that calculates the positional encoding for a transformer:

- max_seq_len is an integer representing the maximum sequence length
- dm is the model depth
- Returns: a numpy.ndarray of shape (max_seq_len, dm) containing the positional encoding vectors
- You should use import numpy as np

```
$ ./4-main.py
(30, 512)
[[ 0.00000000e+00  1.00000000e+00  0.00000000e+00 ...  1.00000000e+00
   0.00000000e+00  1.00000000e+00]
 [ 8.41470985e-01  5.40302306e-01  8.21856190e-01 ...  9.99999994e-01
   1.03663293e-04  9.99999995e-01]
 [ 9.09297427e-01 -4.16146837e-01  9.36414739e-01 ...  9.99999977e-01
   2.07326584e-04  9.99999979e-01]
 ...
 [ 9.56375928e-01 -2.92138809e-01  7.91416314e-01 ...  9.99995791e-01
   2.79890525e-03  9.99996083e-01]
 [ 2.70905788e-01 -9.62605866e-01  9.53248145e-01 ...  9.99995473e-01
   2.90256812e-03  9.99995788e-01]
 [-6.63633884e-01 -7.48057530e-01  2.94705106e-01 ...  9.99995144e-01
   3.00623096e-03  9.99995481e-01]]
$
```

### [4. Scaled Dot Product Attention](./5-sdp_attention.py)

<p>
  <img src="https://holbertonintranet.s3.amazonaws.com/uploads/medias/2020/7/8f5aadef511d9f646f5009756035b472073fe896.png?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIARDDGGGOUWMNL5ANN%2F20200930%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20200930T000755Z&X-Amz-Expires=86400&X-Amz-SignedHeaders=host&X-Amz-Signature=2346c20b72021fbe59410e35a6f482256440cd5e10d3eb0f3a69922024151466">
</p>

Write the function def sdp_attention(Q, K, V, mask=None) that calculates the scaled dot product attention:

- Q is a tensor with its last two dimensions as (..., seq_len_q, dk) containing the query matrix
- K is a tensor with its last two dimensions as (..., seq_len_v, dk) containing the key matrix
- V is a tensor with its last two dimensions as (..., seq_len_v, dv) containing the value matrix
- mask is a tensor that can be broadcast into (..., seq_len_q, seq_len_v) containing the optional mask, or defaulted to None
  - if mask is not None, multiply -1e9 to the mask and add it to the scaled matrix multiplication
- The preceding dimensions of Q, K, and V are the same
- Returns: output, weights
  - outputa tensor with its last two dimensions as (..., seq_len_q, dv) containing the scaled dot product attention
  - weights a tensor with its last two dimensions as (..., seq_len_q, seq_len_v) containing the attention weights

```
$ ./5-main.py
Tensor("MatMul_1:0", shape=(50, 10, 512), dtype=float32)
Tensor("Softmax:0", shape=(50, 10, 15), dtype=float32)
$
```

### [](./)

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)