# 0x01. Clustering

## Learning Objectives

- What is a multimodal distribution?
- What is a cluster?
- What is cluster analysis?
- What is “soft” vs “hard” clustering?
- What is K-means clustering?
- What are mixture models?
- What is a Gaussian Mixture Model (GMM)?
- What is the Expectation-Maximization (EM) algorithm?
- How to implement the EM algorithm for GMMs
- What is cluster variance?
- What is the mountain/elbow method?
- What is the Bayesian Information Criterion?
- How to determine the correct number of clusters
- What is Hierarchical clustering?
- What is Agglomerative clustering?
- What is Ward’s method?
- What is Cophenetic distance?
- What is scikit-learn?
- What is scipy?

## Requirements

- Allowed editors: `vi`, `vim`, `emacs`
- All your files will be interpreted/compiled on Ubuntu 16.04 LTS using `python3` (version 3.5)
- Your files will be executed with `numpy` (version 1.15), `sklearn` (version 0.21), and `scipy` (version 1.3)
- All your files should end with a new line
- The first line of all your files should be exactly `#!/usr/bin/env python3`
- All of your files must be executable
- A `README.md` file, at the root of the folder of the project, is mandatory
- Your code should use the `pycodestyle` style (version 2.4)
- All your modules should have documentation (`python3 -c 'print(__import__("my_module").__doc__)'`)
- All your classes should have documentation (`python3 -c 'print(__import__("my_module").MyClass.__doc__)'`)
- All your functions (inside and outside a class) should have documentation (`python3 -c 'print(__import__("my_module").my_function.__doc__)'` and `python3 -c 'print\
(__import__("my_module").MyClass.my_function.__doc__)'`)
- Unless otherwise noted, you are not allowed to import any module except `import numpy as np`
- Your code should use the minimum number of operations to avoid floating point errors

## Installing Scikit-Learn 0.21.x

```
pip install --user scikit-learn==0.21
```

## Installing Scipy 1.3.x

`scipy` should have already been installed with `matplotlib` and `numpy`, but just in case:

```
pip install --user scipy==1.3
```

## Tasks

### [0. Initialize K-means](./0-initialize.py)

---

### [1. K-means](./1-kmeans.py)

---

### [2. Variance](./2-variance.py)

---

### [3. Optimize k](./3-optimum.py)

---

### [4. Initialize GMM](./4-initialize.py)

---

### [5. PDF](./5-pdf.py)

---

### [6. Expectation](./6-expectation.py)

---

### [7. Maximization](./7-maximization.py)

---

### [8. EM](./8-EM.py)

---

### [9. BIC](./9-BIC.py)

---

### [10. Hello, sklearn!](./10-kmeans.py)

---

### [11. GMM](./11-gmm.py)

---

### [12. Agglomerative](./12-agglomerative.py)

---

## Author

- **Pierre Beaujuge** - [PierreBeaujuge](https://github.com/PierreBeaujuge)