# 0x05. Generative Adversarial Networks

## Some GAN-generated Images in this Project :) (images sampled during training)

<p align='center'>
  <img src='./0x05-images/img_1.png'>
  <img src='./0x05-images/img_2.png'>
  <img src='./0x05-images/img_3.png'>
  <img src='./0x05-images/img_4.png'>
  <img src='./0x05-images/img_5.png'>
</p>

* `Train output`
```
vagrant@ubuntu-xenial:~/holbertonschool-machine_learning/unsupervised_learning/0x05-GANs$ ./5-main.py
Epoch: 0
D_loss: 1.838328242301941
G_loss: 1.8368370532989502
Epoch: 1000
D_loss: 0.026001393795013428
G_loss: 10.522038459777832
Epoch: 2000
D_loss: 0.007320269476622343
G_loss: 8.28874397277832
Epoch: 3000
D_loss: 0.12734414637088776
G_loss: 6.635959625244141
Epoch: 4000
D_loss: 0.10047760605812073
G_loss: 6.356070518493652
Epoch: 5000
D_loss: 0.0552017018198967
G_loss: 5.478587627410889
Epoch: 6000
D_loss: 0.2517339587211609
G_loss: 4.47649621963501
Epoch: 7000
D_loss: 0.22253933548927307
G_loss: 3.5398597717285156
Etc...
```

## Learning Objectives

- What is a generator?
- What is a discriminator?
- What is the minimax loss? modified minimax loss? wasserstein loss?
- How do you train a GAN?
- What are the use cases for GANs?
- What are the shortcoming of GANs?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15) and `tensorflow` (version 1.12)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as keras` and `import numpy as np`, as needed

## Tasks

### [0. Generator](./0-generator.py)

Write a function `def generator(Z):` that creates a simple generator network for MNIST digits:

- `Z` is a `tf.tensor` containing the input to the generator network
- The network should have two layers:
  - the first layer should have 128 nodes and use relu activation with name `layer_1`
  - the second layer should have 784 nodes and use a sigmoid activation with name `layer_2`
- All variables in the network should have the scope `generator` with `reuse=tf.AUTO_REUSE`
- Returns `X`, a `tf.tensor` containing the generated image

---

### [1. Discriminator](./1-discriminator.py)

Write a function `def discriminator(X):` that creates a discriminator network for MNIST digits:

- `X` is a `tf.tensor` containing the input to the discriminator network
- The network should have two layers:
  - the first layer should have 128 nodes and use relu activation with name `layer_1`
  - the second layer should have 1 node and use a sigmoid activation with name `layer_2`
- All variables in the network should have the scope `discriminator` with `reuse=tf.AUTO_REUSE`
- Returns `Y`, a `tf.tensor` containing the classification made by the discriminator

---

### [2. Train Discriminator](./2-train_discriminator.py)

Write a function def `train_discriminator(Z, X):` that creates the loss tensor and training op for the discriminator:

- `Z` is the `tf.placeholder` that is the input for the generator
- `X` is the `tf.placeholder` that is the real input for the discriminator
- You can use the following imports:
  - `generator = __import__('0-generator').generator`
  - `discriminator = __import__('1-discriminator').discriminator`
- The discriminator should minimize the negative minimax loss
- The discriminator should be trained using Adam optimization
- The generator should NOT be trained
- Returns: `loss, train_op`
  - `loss` is the discriminator loss
  - `train_op` is the training operation for the discriminator

---

### [3. Train Generator](./3-train_generator.py)

Write a function def train_generator(Z): that creates the loss tensor and training op for the generator:

- `Z` is the `tf.placeholder` that is the input for the generator
- `X` is the `tf.placeholder` that is the input for the discriminator
- You can use the following imports:
  - `generator = __import__('0-generator').generator`
  - `discriminator = __import__('1-discriminator').discriminator`
- The generator should minimize the negative modified minimax loss
- The generator should be trained using Adam optimization
- The discriminator should NOT be trained
- Returns: `loss, train_op`
  - `loss` is the generator loss
  - `train_op` is the training operation for the generator

---

### [4. Sample Z](./4-sample_Z.py)

Write a function `def sample_Z(m, n):` that creates input for the generator:

- `m` is the number of samples that should be generated
- `n` is the number of dimensions of each sample
- All samples should be taken from a random uniform distribution within the range `[-1, 1]`
- Returns: `Z`, a `numpy.ndarray` of shape `(m, n)` containing the uniform samples

---

### [5. Train GAN](./5-train_GAN.py)

Write a function def train_gan(X, epochs, batch_size, Z_dim, save_path='/tmp'): that trains a GAN:

- X is a np.ndarray of shape (m, 784) containing the real data input
  - m is the number of real data samples
- epochs is the number of epochs that the each network should be trained for
- batch_size is the batch size that should be used during training
- Z_dim is the number of dimensions for the randomly generated input
- save_path is the path to save the trained generator
  - Create the tf.placeholder for Z and add it to the graph’s collection
- The discriminator and generator training should be altered after one epoch
- You can use the following imports:
  - train_generator = __import__('2-train_generator').train_generator
  - train_discriminator = __import__('3-train_discriminator').train_discriminator
  - sample_Z = __import__('4-sample_Z').sample_Z

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)