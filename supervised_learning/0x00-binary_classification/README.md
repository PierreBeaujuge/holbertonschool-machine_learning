# 0x00. Binary Classification

## Learning Objectives

- What is a model?
- What is supervised learning?
- What is a prediction?
- What is a node?
- What is a weight?
- What is a bias?
- What are activation functions?
  - Sigmoid?
  - Tanh?
  - Relu?
  - Softmax?
- What is a layer?
- What is a hidden layer?
- What is Logistic Regression?
- What is a loss function?
- What is a cost function?
- What is forward propagation?
- What is Gradient Descent?
- What is back propagation?
- What is a Computation Graph?
- How to initialize weights/biases
- The importance of vectorization
- How to split up your data

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise stated, you cannot import any module except `import numpy as np`
- Unless otherwise noted, you are not allowed to use any loops (for, while, etc.)

## Testing your code

In order to test your code, you’ll need DATA! Please download these datasets (Binary_Train.npz, Binary_Dev.npz) to go along with all of the following main files. You do not need to upload these files to GitHub. Your code will not necessarily be tested with these datasets. All of the following code assumes that you have stored all of your datasets in a separate `data` directory.

```
alexa@ubuntu-xenial:0x00-binary_classification$ cat show_data.py
```
```py
#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np

lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']

fig = plt.figure(figsize=(10, 10))
for i in range(100):
    fig.add_subplot(10, 10, i + 1)
    plt.imshow(X_3D[i])
    plt.title(Y[0, i])
    plt.axis('off')
plt.tight_layout()
plt.show()
```
```
alexa@ubuntu-xenial:0x00-binary_classification$ ./show_data.py
```

## Tasks

### [0. Neuron](./0-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification:

- class constructor: `def __init__(self, nx):`
  - `nx` is the number of input features to the neuron
    - If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be an integer`
    -  If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be a positive integer`
  - All exceptions should be raised in the order listed above
- Public instance attributes:
  - `W`: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `b`: The bias for the neuron. Upon instantiation, it should be initialized to 0.
  - `A`: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./0-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
(1, 784)
0
0
10
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [1. Privatize Neuron](./1-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `0-neuron.py`):

- class constructor: `def __init__(self, nx):`
  - `nx` is the number of input features to the neuron
    - If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be a integer`
    - If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be positive`
  - All exceptions should be raised in the order listed above
- **Private** instance attributes:
  - `__W`: The weights vector for the neuron. Upon instantiation, it should be initialized using a random normal distribution.
  - `__b`: The bias for the neuron. Upon instantiation, it should be initialized to 0.
  - `__A`: The activated output of the neuron (prediction). Upon instantiation, it should be initialized to 0.
  - Each private attribute should have a corresponding getter function (no setter function).

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./1-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0
0
Traceback (most recent call last):
File "./1-main.py", line 16, in <module>
neuron.A = 10
AttributeError: can't set attribute
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [2. Neuron Forward Propagation](./2-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `1-neuron.py`):

- Add the public method `def forward_prop(self, X):`
  - Calculates the forward propagation of the neuron
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - Updates the private attribute `__A`
  - The neuron should use a sigmoid activation function
  - Returns the private attribute `__A`

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./2-main.py
[[5.34775247e-10 7.24627778e-04 4.52416436e-07 ... 8.75691930e-05
1.13141966e-06 6.55799932e-01]]
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [3. Neuron Cost ](./3-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `2-neuron.py`):

- Add the public method `def cost(self, Y, A):`
  - Calculates the cost of the model using logistic regression
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A` is a `numpy.ndarray` with shape (1, `m`) containing the activated output of the neuron for each example
  - To avoid division by zero errors, please use `1.0000001 - A` instead of `1 - A`
  - Returns the cost

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./3-main.py
4.365104944262272
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [4. Evaluate Neuron](./4-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `3-neuron.py`):

- Add the public method `def evaluate(self, X, Y):`
  - Evaluates the neuron’s predictions
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - Returns the neuron’s prediction and the cost of the network, respectively
    - The prediction should be a `numpy.ndarray` with shape (1, `m`) containing the predicted labels for each example
    - The label values should be 1 if the output of the network is >= 0.5 and 0 otherwise

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./4-main.py
[[0 0 0 ... 0 0 0]]
4.365104944262272
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [5. Neuron Gradient Descent](./5-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `4-neuron.py`):

- Add the public method `def gradient_descent(self, X, Y, A, alpha=0.05):`
  - Calculates one pass of gradient descent on the neuron
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `A` is a `numpy.ndarray` with shape (1, `m`) containing the activated output of the neuron for each example
  - `alpha` is the learning rate
  - Updates the private attributes `__W` and `__b`

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./5-main.py
[[ 1.76405235e+00  4.00157208e-01  9.78737984e-01  2.24089320e+00
   1.86755799e+00 -9.77277880e-01  9.50088418e-01 -1.51357208e-01

...

  -5.85865511e-02 -3.17543094e-01 -1.63242330e+00 -6.71341546e-02
   1.48935596e+00  5.21303748e-01  6.11927193e-01 -1.34149673e+00]]
0.2579495783615682
alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [6. Train Neuron](./6-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `5-neuron.py`):

- Add the public method `def train(self, X, Y, iterations=5000, alpha=0.05):`
  - Trains the neuron
  - `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
    - `nx` is the number of input features to the neuron
    - `m` is the number of examples
  - `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
  - `iterations` is the number of iterations to train over
    - if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
    - if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
  - `alpha` is the learning rate
    - if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
    - if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
  - All exceptions should be raised in the order listed above
  - Updates the private attributes `__W`, `__b`, and `__A`
  - You are allowed to use one loop
  - Returns the evaluation of the training data after `iterations` of training have occurred

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./6-main.py
Train cost: 1.3805076999077135
Train accuracy: 64.73746545598105%
Dev cost: 1.4096194345468178
Dev accuracy: 64.49172576832152%
```
_Not that great… Let’s get more data!_

---

### [7. Upgrade Train Neuron](./7-neuron.py)

Write a class `Neuron` that defines a single neuron performing binary classification (Based on `6-neuron.py`):

*   Update the public method `train` to `def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):`
    *   Trains the neuron by updating the private attributes `__W`, `__b`, and `__A`
    *   `X` is a `numpy.ndarray` with shape (`nx`, `m`) that contains the input data
        *   `nx` is the number of input features to the neuron
        *   `m` is the number of examples
    *   `Y` is a `numpy.ndarray` with shape (1, `m`) that contains the correct labels for the input data
    *   `iterations` is the number of iterations to train over
        *   if `iterations` is not an integer, raise a `TypeError` with the exception `iterations must be an integer`
        *   if `iterations` is not positive, raise a `ValueError` with the exception `iterations must be a positive integer`
    *   `alpha` is the learning rate
        *   if `alpha` is not a float, raise a `TypeError` with the exception `alpha must be a float`
        *   if `alpha` is not positive, raise a `ValueError` with the exception `alpha must be positive`
    *   `verbose` is a boolean that defines whether or not to print information about the training. If `True`, print `Cost after {iteration} iterations: {cost}` every `step` iterations:
        *   Include data from the 0th and last iteration
    *   `graph` is a boolean that defines whether or not to graph information about the training once the training has completed. If `True`:
        *   Plot the training data every `step` iterations as a blue line
        *   Label the x-axis as `iteration`
        *   Label the y-axis as `cost`
        *   Title the plot `Training Cost`
        *   Include data from the 0th and last iteration
    *   Only if either `verbose` or `graph` are `True`:
        *   if `step` is not an integer, raise a `TypeError` with the exception `step must be an integer`
        *   if `step` is not positive or is greater than `iterations`, raise a `ValueError` with the exception `step must be positive and <= iterations`
    *   All exceptions should be raised in the order listed above
    *   The 0th iteration should represent the state of the neuron before any training has occurred
    *   You are allowed to use one loop
    *   You can use `import matplotlib.pyplot as plt`
    *   Returns: the evaluation of the training data after `iterations` of training have occurred

```
alexa@ubuntu-xenial:0x00-binary_classification$ ./7-main.py
Cost after 0 iterations: 4.365104944262272
Cost after 100 iterations: 0.11955134491351888

...

Cost after 3000 iterations: 0.013386353289868338
```

---

### [8. NeuralNetwork](./8-neural_network.py)

Write a class `NeuralNetwork` that defines a neural network with one hidden layer performing binary classification:

*   class constructor: `def __init__(self, nx, nodes):`
    *   `nx` is the number of input features
        *   If `nx` is not an integer, raise a `TypeError` with the exception: `nx must be an integer`
        *   If `nx` is less than 1, raise a `ValueError` with the exception: `nx must be a positive integer`
    *   `nodes` is the number of nodes found in the hidden layer
        *   If `nodes` is not an integer, raise a `TypeError` with the exception: `nodes must be an integer`
        *   If `nodes` is less than 1, raise a `ValueError` with the exception: `nodes must be a positive integer`
    *   All exceptions should be raised in the order listed above
*   Public instance attributes:
    *   `W1`: The weights vector for the hidden layer. Upon instantiation, it should be initialized using a random normal distribution.
    *   `b1`: The bias for the hidden layer. Upon instantiation, it should be initialized with 0’s.
    *   `A1`: The activated output for the hidden layer. Upon instantiation, it should be initialized to 0.
    *   `W2`: The weights vector for the output neuron. Upon instantiation, it should be initialized using a random normal distribution.
    *   `b2`: The bias for the output neuron. Upon instantiation, it should be initialized to 0.
    *   `A2`: The activated output for the output neuron (prediction). Upon instantiation, it should be initialized to 0.

```
    alexa@ubuntu-xenial:0x00-binary_classification$ ./8-main.py
    [[ 1.76405235  0.40015721  0.97873798 ...  0.52130375  0.61192719
      -1.34149673]
     [ 0.47689837  0.14844958  0.52904524 ...  0.0960042  -0.0451133
       0.07912172]
     [ 0.85053068 -0.83912419 -1.01177408 ... -0.07223876  0.31112445
      -1.07836109]]
    (3, 784)
    [[0.]
     [0.]
     [0.]]
    [[ 1.06160017 -1.18488744 -1.80525169]]
    (1, 3)
    0
    0
    0
    10
    alexa@ubuntu-xenial:0x00-binary_classification$
```

---

### [9. Privatize NeuralNetwork](./9-neural_network.py)

---

### [10. NeuralNetwork Forward Propagation](./10-neural_network.py)

---

### [11. NeuralNetwork Cost](./11-neural_network.py)

---

### [12. Evaluate NeuralNetwork](./12-neural_network.py)

---

### [13. NeuralNetwork Gradient Descent](./13-neural_network.py)

---

### [14. Train NeuralNetwork](./14-neural_network.py)

---

### [15. Upgrade Train NeuralNetwork](./15-neural_network.py)

---

### [16. DeepNeuralNetwork](./16-deep_neural_network.py)

---

### [17. Privatize DeepNeuralNetwork](./17-deep_neural_network.py)

---

### [18. DeepNeuralNetwork Forward Propagation](./18-deep_neural_network.py)

---

### [19. DeepNeuralNetwork Cost](./19-deep_neural_network.py)

---

### [20. Evaluate DeepNeuralNetwork](./20-deep_neural_network.py)

---

### [21. DeepNeuralNetwork Gradient Descent](./21-deep_neural_network.py)

---

### [22. Train DeepNeuralNetwork](./22-deep_neural_network.py)

---

### [23. Upgrade Train DeepNeuralNetwork](./23-deep_neural_network.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)