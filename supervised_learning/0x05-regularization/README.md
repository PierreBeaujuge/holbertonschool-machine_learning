# 0x05. Regularization

## Learning Objectives

- What is regularization? What is its purpose?
- What is are L1 and L2 regularization? What is the difference between the two methods?
- What is dropout?
- What is early stopping?
- What is data augmentation?
- How do you implement the above regularization methods in Numpy? Tensorflow?
- What are the pros and cons of the above regularization methods?

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
- Unless otherwise noted, you are not allowed to import any module except import `numpy as np` and `import tensorflow as tf`
- You are not allowed to use the keras module in tensorflow
- You should not import any module unless it is being used
- When initializing layer weights, use `tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")`

## Tasks

### [0. L2 Regularization Cost](./0-l2_reg_cost.py)

---

### [1. Gradient Descent with L2 Regularization](./1-l2_reg_gradient_descent.py)

---

### [2. L2 Regularization Cost](./2-l2_reg_cost.py)

---

### [3. Create a Layer with L2 Regularization](./3-l2_reg_create_layer.py)

---

### [4. Forward Propagation with Dropout](./4-dropout_forward_prop.py)

---

### [5. Gradient Descent with Dropout](./5-dropout_gradient_descent.py)

---

### [6. Create a Layer with Dropout](./6-dropout_create_layer.py)

---

### [7. Early Stopping](./7-early_stopping.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)