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
- Your files will be executed with `numpy` (version 1.16)
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

---

### [1. Privatize Neuron](./1-neuron.py)

---

### [2. Neuron Forward Propagation](./2-neuron.py)

---

### [3. Neuron Cost ](./3-neuron.py)

---

### [4. Evaluate Neuron](./4-neuron.py)

---

### [5. Neuron Gradient Descent](./5-neuron.py)

---

### [6. Train Neuron](./6-neuron.py)

---

### [7. Upgrade Train Neuron](./7-neuron.py)

---

### [8. NeuralNetwork](./8-neural_network.py)

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