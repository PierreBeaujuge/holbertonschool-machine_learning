# 0x02. Tensorflow

## Learning Objectives

- What is tensorflow?
- What is a session? graph?
- What are tensors?
- What are variables? constants? placeholders? How do you use them?
- What are operations? How do you use them?
- What are namespaces? How do you use them?
- How to train a neural network in tensorflow
- What is a checkpoint?
- How to save/load a model with tensorflow
- What is the graph collection?
- How to add and get variables from the collection

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
- Unless otherwise noted, you are not allowed to import any module except `import tensorflow as tf`
- You are not allowed to use the `keras` module in `tensorflow`

## Installing Tensorflow 1.12

```
$ pip install --user tensorflow==1.12
```

## Tasks

### [0. Placeholders](./0-create_placeholders.py)

Write the function `def create_placeholders(nx, classes):` that returns two placeholders, `x` and `y`, for the neural network:

*   `nx`: the number of feature columns in our data
*   `classes`: the number of classes in our classifier
*   Returns: placeholders named `x` and `y`, respectively
    *   `x` is the placeholder for the input data to the neural network
    *   `y` is the placeholder for the one-hot labels for the input data

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./0-main.py 
    Tensor("x:0", shape=(?, 784), dtype=float32)
    Tensor("y:0", shape=(?, 10), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

---

### [1. Layers](./1-create_layer.py)

Write the function `def create_layer(prev, n, activation):`

*   `prev` is the tensor output of the previous layer
*   `n` is the number of nodes in the layer to create
*   `activation` is the activation function that the layer should use
*   use `tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")` to implement `He et. al` initialization for the layer weights
*   each layer should be given the name `layer`
*   Returns: the tensor output of the layer

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./1-main.py 
    Tensor("layer/Tanh:0", shape=(?, 256), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 
 ```

---

### [2. Forward Propagation](./2-forward_prop.py)

Write the function `def forward_prop(x, layer_sizes=[], activations=[]):` that creates the forward propagation graph for the neural network:

*   `x` is the placeholder for the input data
*   `layer_sizes` is a list containing the number of nodes in each layer of the network
*   `activations` is a list containing the activation functions for each layer of the network
*   Returns: the prediction of the network in tensor form
*   For this function, you should import your `create_layer` function with `create_layer = __import__('1-create_layer').create_layer`

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./2-main.py 
    Tensor("layer_2/BiasAdd:0", shape=(?, 10), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

---

### [3. Accuracy](./3-calculate_accuracy.py)

Write the function `def calculate_accuracy(y, y_pred):` that calculates the accuracy of a prediction:

*   `y` is a placeholder for the labels of the input data
*   `y_pred` is a tensor containing the network’s predictions
*   Returns: a tensor containing the decimal accuracy of the prediction

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ cat 3-main.py 
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./3-main.py 
    Tensor("Mean:0", shape=(), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$
```

---

### [4. Loss](./4-calculate_loss.py)

Write the function `def calculate_loss(y, y_pred):` that calculates the softmax cross-entropy loss of a prediction:

*   `y` is a placeholder for the labels of the input data
*   `y_pred` is a tensor containing the network’s predictions
*   Returns: a tensor containing the loss of the prediction

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./4-main.py 
    Tensor("softmax_cross_entropy_loss/value:0", shape=(), dtype=float32)
    ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

---

### [5. Train_Op](./5-create_train_op.py)

Write the function `def create_train_op(loss, alpha):` that creates the training operation for the network:

*   `loss` is the loss of the network’s prediction
*   `alpha` is the learning rate
*   Returns: an operation that trains the network using gradient descent

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./5-main.py 
    name: "GradientDescent"
    op: "NoOp"
    input: "^GradientDescent/update_layer/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer/bias/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_1/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_1/bias/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_2/kernel/ApplyGradientDescent"
    input: "^GradientDescent/update_layer_2/bias/ApplyGradientDescent"
    
    ubuntu@alexa-ml:~/0x02-tensorflow$ 
```

---

### [6. Train](./6-train.py)

Write the function `def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations, alpha, iterations, save_path="/tmp/model.ckpt"):` that builds, trains, and saves a neural network classifier:

*   `X_train` is a `numpy.ndarray` containing the training input data
*   `Y_train` is a `numpy.ndarray` containing the training labels
*   `X_valid` is a `numpy.ndarray` containing the validation input data
*   `Y_valid` is a `numpy.ndarray` containing the validation labels
*   `layer_sizes` is a list containing the number of nodes in each layer of the network
*   `actications` is a list containing the activation functions for each layer of the network
*   `alpha` is the learning rate
*   `iterations` is the number of iterations to train over
*   `save_path` designates where to save the model
*   Add the following to the graph’s collection
    *   placeholders `x` and `y`
    *   tensors `y_pred`, `loss`, and `accuracy`
    *   operation `train_op`
*   After every 100 iterations, the 0th iteration, and `iterations` iterations, print the following:
    *   `After {i} iterations:` where i is the iteration
    *   `\tTraining Cost: {cost}` where `{cost}` is the training cost
    *   `\tTraining Accuracy: {accuracy}` where `{accuracy}` is the training accuracy
    *   `\tValidation Cost: {cost}` where `{cost}` is the validation cost
    *   `\tValidation Accuracy: {accuracy}` where `{accuracy}` is the validation accuracy
*   _Reminder: the 0th iteration represents the model before any training has occurred_
*   After training has completed, save the model to `save_path`
*   You may use the following imports:
    *   `calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy`
    *   `calculate_loss = __import__('4-calculate_loss').calculate_loss`
    *   `create_placeholders = __import__('0-create_placeholders').create_placeholders`
    *   `create_train_op = __import__('5-create_train_op').create_train_op`
    *   `forward_prop = __import__('2-forward_prop').forward_prop`
*   You are not allowed to use `tf.saved_model`
*   Returns: the path where the model was saved

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./6-main.py 
    2018-11-03 01:04:55.281078: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
    After 0 iterations:
        Training Cost: 2.8232274055480957
        Training Accuracy: 0.08726000040769577
        Validation Cost: 2.810533285140991
        Validation Accuracy: 0.08640000224113464
    After 100 iterations:
        Training Cost: 0.8393384218215942
        Training Accuracy: 0.7824000120162964
        Validation Cost: 0.7826032042503357
        Validation Accuracy: 0.8061000108718872
    After 200 iterations:
        Training Cost: 0.6094841361045837
        Training Accuracy: 0.8396000266075134
        Validation Cost: 0.5562412142753601
        Validation Accuracy: 0.8597999811172485
    
    ...
    
    After 1000 iterations:
        Training Cost: 0.352960467338562
        Training Accuracy: 0.9004999995231628
        Validation Cost: 0.32148978114128113
        Validation Accuracy: 0.909600019454956
    Model saved in path: ./model.ckpt
    ubuntu@alexa-ml:~/0x02-tensorflow$ ls model.ckpt*
    model.ckpt.data-00000-of-00001  model.ckpt.index  model.ckpt.meta
    ubuntu@alexa-ml:~/0x02-tensorflow$
```

---

### [7. Evaluate](./7-evaluate.py)

Write the function `def evaluate(X, Y, save_path):` that evaluates the output of a neural network:

*   `X` is a `numpy.ndarray` containing the input data to evaluate
*   `Y` is a `numpy.ndarray` containing the one-hot labels for `X`
*   `save_path` is the location to load the model from
*   You are not allowed to use `tf.saved_model`
*   Returns: the network’s prediction, accuracy, and loss, respectively

```
    ubuntu@alexa-ml:~/0x02-tensorflow$ ./7-main.py
    2018-11-03 02:08:30.767168: I tensorflow/core/common_runtime/process_util.cc:69] Creating new thread pool with default inter op setting: 2. Tune using inter_op_parallelism_threads for best performance.
    Test Accuracy: 0.9391
    Test Cost: 0.21756475
```

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)