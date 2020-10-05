# 0x12. Transformer Applications

<p align="center">
  <img src='./0x12-images/img_1.png'>
</p>

## Learning Objectives

- How to use Transformers for Machine Translation
- How to write a custom train/test loop in Keras
- How to use Tensorflow Datasets

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.16) and `tensorflow` (version 1.15)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise stated, you cannot import any module except `import tensorflow.compat.v2 as tf`

## TF Datasets

For machine translation, we will be using the prepared Tensorflow Datasets ted_hrlr_translate/pt_to_en for English to Portuguese translation

To download Tensorflow Datasets, please use:

```
pip install --user tensorflow-datasets
```

To use this dataset, we will have to use the Tensorflow 2.0 compat within Tensorflow 1.15 and download the content:

```
$ cat download_tfdataset.py
#!/usr/bin/env python3
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

pt2en_train = tfds.load('ted_hrlr_translate/pt_to_en', split='train', as_supervised=True)
for pt, en in pt2en_train.take(1):
  print(pt.numpy().decode('utf-8'))
  print(en.numpy().decode('utf-8'))
$ ./download_tfdataset.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
$
```

## Tasks

### [0. Dataset](./0-dataset.py)

Create the class `Dataset` that loads and preps a dataset for machine translation:

- Class constructor `def __init__(self):`
  - creates the instance attributes:
    - `data_train`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `train` split, loaded `as_supervided`
    - `data_valid`, which contains the `ted_hrlr_translate/pt_to_en` `tf.data.Dataset` `validate` split, loaded `as_supervided`
    - `tokenizer_pt` is the Portuguese tokenizer created from the training set
    - `tokenizer_en` is the English tokenizer created from the training set
- Create the instance method `def tokenize_dataset(self, data):` that creates sub-word tokenizers for our dataset:
  - `data` is a `tf.data.Dataset` whose examples are formatted as a tuple `(pt, en)`
    - `pt` is the `tf.Tensor` containing the Portuguese sentence
    - `en` is the `tf.Tensor` containing the corresponding English sentence
  - The maximum vocab size should be set to `2**15`
  - Returns: `tokenizer_pt, tokenizer_en`
    - `tokenizer_pt` is the Portuguese tokenizer
    - `tokenizer_en` is the English tokenizer

```
$ ./0-main.py
e quando melhoramos a procura , tiramos a única vantagem da impressão , que é a serendipidade .
and when you improve searchability , you actually take away the one advantage of print , which is serendipity .
tinham comido peixe com batatas fritas ?
did they eat fish and chips ?
<class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'>
<class 'tensorflow_datasets.core.features.text.subword_text_encoder.SubwordTextEncoder'>
$
```

---

### [1. Encode Tokens](./1-dataset.py)

Update the class `Dataset`:

- Create the instance method `def encode(self, pt, en):` that encodes a translation into tokens:
  - `pt` is the `tf.Tensor` containing the Portuguese sentence
  - `en` is the `tf.Tensor` containing the corresponding English sentence
  - The tokenized sentences should include the start and end of sentence tokens
  - The start token should be indexed as `vocab_size`
  - The end token should be indexed as `vocab_size + 1`
  - Returns: `pt_tokens, en_tokens`
    - `pt_tokens` is a `tf.Tensor` containing the Portuguese tokens
    - `en_tokens` is a `tf.Tensor` containing the English tokens

```
$ ./1-main.py
([30138, 6, 36, 17925, 13, 3, 3037, 1, 4880, 3, 387, 2832, 18, 18444, 1, 5, 8, 3, 16679, 19460, 739, 2, 30139], [28543, 4, 56, 15, 1266, 20397, 10721, 1, 15, 100, 125, 352, 3, 45, 3066, 6, 8004, 1, 88, 13, 14859, 2, 28544])
([30138, 289, 15409, 2591, 19, 20318, 26024, 29997, 28, 30139], [28543, 93, 25, 907, 1366, 4, 5742, 33, 28544])
$
```

---

### [2. TF Encode](./2-dataset.py)

Update the class `Dataset`:

- Create the instance method `def tf_encode(self, pt, en):` that acts as a `tensorflow` wrapper for the `encode` instance method
  - Make sure to set the shape of the `pt` and `en` return tensors
- Update the class constructor `def __init__(self):`
  - update the `data_train` and `data_validate` attributes by tokenizing the examples

```
$ ./2-main.py
tf.Tensor(
[30138     6    36 17925    13     3  3037     1  4880     3   387  2832
    18 18444     1     5     8     3 16679 19460   739     2 30139], shape=(23,), dtype=int64) tf.Tensor(
[28543     4    56    15  1266 20397 10721     1    15   100   125   352
    3    45  3066     6  8004     1    88    13 14859     2 28544], shape=(23,), dtype=int64)
tf.Tensor([30138   289 15409  2591    19 20318 26024 29997    28 30139], shape=(10,), dtype=int64) tf.Tensor([28543    93    25   907  1366     4  5742    33 28544], shape=(9,), dtype=int64)
$
```

---

### [3. Pipeline](./3-dataset.py)

Update the class `Dataset` to set up the data pipeline:

- Update the class constructor `def __init__(self, batch_size, max_len)`:
  - `batch_size` is the batch size for training/validation
  - `max_len` is the maximum number of tokens allowed per example sentence
  - update the `data_train` attribute by performing the following actions:
    - filter out all examples that have either sentence with more than `max_len` tokens
    - cache the dataset to increase performance
    - shuffle the entire dataset
    - split the dataset into padded batches of size `batch_size`
    - prefetch the dataset using `tf.data.experimental.AUTOTUNE` to increase performance
  - update the `data_validate` attribute by performing the following actions:
    - filter out all examples that have either sentence with more than `max_len` tokens
    - split the dataset into padded batches of size `batch_size`

```
$ ./3-main.py
tf.Tensor(
[[30138  1029   104 ...     0     0     0]
 [30138    40     8 ...     0     0     0]
 [30138    12    14 ...     0     0     0]
 ...
 [30138    72 23483 ...     0     0     0]
 [30138  2381   420 ...     0     0     0]
 [30138     7 14093 ...     0     0     0]], shape=(32, 39), dtype=int64) tf.Tensor(
[[28543   831   142 ...     0     0     0]
 [28543    16    13 ...     0     0     0]
 [28543    19     8 ...     0     0     0]
 ...
 [28543    18    27 ...     0     0     0]
 [28543  2648   114 ... 28544     0     0]
 [28543  9100 19214 ...     0     0     0]], shape=(32, 37), dtype=int64)
tf.Tensor([30138   289 15409  2591    19 20318 26024 29997    28 30139], shape=(10,), dtype=int64) tf.Tensor([28543    93    25   907  1366     4  5742    33 28544], shape=(9,), dtype=int64)
$
```

---

### [4. Create Masks](./4-create_masks.py)

Create the function `def create_masks(inputs, target):` that creates all masks for training/validation:

- `inputs` is a `tf.Tensor` of shape `(batch_size, seq_len_in)` that contains the input sentence
- `target` is a `tf.Tensor` of shape `(batch_size, seq_len_out)` that contains the target sentence
- This function should only use tensorflow operations in order to properly function in the training step
- Returns: `encoder_mask, look_ahead_mask, decoder_mask`
  - `encoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the encoder
  - `look_ahead_mask` is the `tf.Tensor` look ahead mask of shape `(batch_size, 1, seq_len_out, seq_len_out)` to be applied in the decoder
  - `decoder_mask` is the `tf.Tensor` padding mask of shape `(batch_size, 1, 1, seq_len_in)` to be applied in the decoder

```
$ ./4-main.py
(<tf.Tensor: id=414557, shape=(32, 1, 1, 39), dtype=float32, numpy=
array([[[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       ...,
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: id=414573, shape=(32, 1, 1, 39), dtype=float32, numpy=
       array([[[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       ...,
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]],
       [[[0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>, <tf.Tensor: id=414589, shape=(32, 1, 37, 37), dtype=float32, numpy=
array([[[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.]]],

       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.]]],

       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.]]],

       ...,

       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.]]],

       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 0., 1., 1.],
	 [0., 0., 0., ..., 0., 1., 1.],
	 [0., 0., 0., ..., 0., 1., 1.]]],

       [[[0., 1., 1., ..., 1., 1., 1.],
         [0., 0., 1., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 ...,
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.],
	 [0., 0., 0., ..., 1., 1., 1.]]]], dtype=float32)>)
$
```

---

### [5. Train](./5-train.py)

Take your implementation of a transformer from our previous project and save it to the file `5-transformer.py`. Note, you may need to make slight adjustments to this model to get it to functionally train.

Write a the function `def train_transformer(N, dm, h, hidden, max_len, batch_size, epochs):` that creates and trains a transformer model for machine translation of Portuguese to English using our previously created dataset:

- `N` the number of blocks in the encoder and decoder
- `dm` the dimensionality of the model
- `h` the number of heads
- `hidden` the number of hidden units in the fully connected layers
- `max_len` the maximum number of tokens per sequence
- `batch_size` the batch size for training
- `epochs` the number of epochs to train for
- You should use the following imports:
  - `Dataset = __import__('3-dataset').Dataset`
  - `create_masks = __import__('4-create_masks').create_masks`
  - `Transformer = __import__('5-transformer').Transformer`
- Your model should be trained with Adam optimization and sparse categorical crossentropy
- Your model should show the metrics
- Your model should print information about the training every 50 batches and every epoch, formatted as shown in the example below
- Returns the trained model

```
$ ./5-main.py
Epoch 1, batch 50: loss 7.090165615081787 accuracy 0.03189198300242424
Epoch 1, batch 100: loss 6.863615989685059 accuracy 0.03287176787853241
Epoch 1, batch 150: loss 6.790400505065918 accuracy 0.033233970403671265
Epoch 1, batch 200: loss 6.74777364730835 accuracy 0.03339668735861778
Epoch 1, batch 250: loss 6.725172519683838 accuracy 0.033377211540937424
Epoch 1, batch 300: loss 6.700491905212402 accuracy 0.03350349888205528
Epoch 1, batch 350: loss 6.682713985443115 accuracy 0.03359624370932579

...

$
```

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)