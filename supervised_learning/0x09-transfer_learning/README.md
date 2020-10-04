# 0x09. Transfer Learning

## This project was summarized in a Medium article (follow the link!):
### ["Classifying Images from the CIFAR10 Dataset with Pre-Trained CNNs Using Transfer Learning"](https://medium.com/@pierre.beaujuge/classifying-images-from-the-cifar10-dataset-with-pre-trained-cnns-using-transfer-learning-9348f6d878a8)

## Learning Objectives

- What is a transfer learning?
- What is fine-tuning?
- What is a frozen layer? How and why do you freeze a layer?
- How to use transfer learning with Keras applications

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15) and tensorflow (version 1.12)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow.keras as K`

## Tasks

### [0. Transfer Knowledge](./0-transfer.py)

Write a python script that trains a convolutional neural network to classify the CIFAR 10 dataset:

*   You must use one of the applications listed in [Keras Applications](/rltoken/x6jAoAGkY9dHNZwT-uenow "Keras Applications")
*   Your script must save your trained model in the current working directory as `cifar10.h5`
*   Your saved model should be compiled
*   Your saved model should have a validation accuracy of 88% or higher
*   Your script should not run when the file is imported
*   _Hint: The training may take a while, start early!_

In the same file, write a function `def preprocess_data(X, Y):` that pre-processes the data for your model:

*   `X` is a `numpy.ndarray` of shape `(m, 32, 32, 3)` containing the CIFAR 10 data, where m is the number of data points
*   `Y` is a `numpy.ndarray` of shape `(m,)` containing the CIFAR 10 labels for `X`
*   Returns: `X_p, Y_p`
    *   `X_p` is a `numpy.ndarray` containing the preprocessed `X`
    *   `Y_p` is a `numpy.ndarray` containing the preprocessed `Y`

```
    alexa@ubuntu-xenial:0x09-transfer_learning$ ./0-main.py
    10000/10000 [==============================] - 159s 16ms/sample - loss: 0.3329 - acc: 0.8864
```

---

### [1. Medium Article](https://medium.com/@pierre.beaujuge/classifying-images-from-the-cifar10-dataset-with-pre-trained-cnns-using-transfer-learning-9348f6d878a8)

Write a blog post explaining your experimental process in completing the task above written as a journal-style scientific paper.

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)