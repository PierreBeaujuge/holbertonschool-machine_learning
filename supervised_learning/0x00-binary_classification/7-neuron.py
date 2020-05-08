#!/usr/bin/env python3
"""
Binary Classification
"""
import numpy as np
import matplotlib.pyplot as plt


class Neuron:
    """
    define the Neuron class
    """

    def __init__(self, nx):
        """initialize variables and methods"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        self.nx = nx
        self.__W = np.random.normal(loc=0.0, scale=1.0, size=nx).reshape(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """getter for W"""
        return self.__W

    @property
    def b(self):
        """getter for b"""
        return self.__b

    @property
    def A(self):
        """getter for A"""
        return self.__A

    def forward_prop(self, X):
        """forward propagation function"""
        Z = np.matmul(self.W, X) + self.b
        self.__A = self.sigmoid(Z)
        return self.A

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * Y))

    def cost(self, Y, A):
        """defnine the cost function"""
        m = Y.shape[1]
        return (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """function that evaluates the neuron's predictions"""
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """function that calculates one pass of gradient descent"""
        dZ = A - Y
        m = Y.shape[1]
        dW = (1 / m) * np.matmul(dZ, X.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
        self.__W -= alpha * dW
        self.__b -= (alpha * db)[0][0]

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """function that trains the neuron"""
        if not isinstance(iterations, int):
            raise TypeError('iterations must be an integer')
        if iterations <= 0:
            raise ValueError('iterations must be a positive integer')
        if not isinstance(alpha, float):
            raise TypeError('alpha must be a float')
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if verbose is True or graph is True:
            if not isinstance(step, int):
                raise TypeError('step must be an integer')
            if step <= 0 or step > iterations:
                raise ValueError('step must be positive and <= iterations')
        cost_data = []
        step_data = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            # backpropagate except for last iteration (3000):
            if i != iterations:
                self.gradient_descent(X, Y, A, alpha)
            if (i % step) == 0:
                cost = self.cost(Y, A)
                cost_data += [cost]
                step_data += [i]
                if verbose is True:
                    print('Cost after {} iterations: {}'.format(i, cost))
        if graph is True:
            plt.plot(step_data, cost_data, 'b')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()
        return np.where(A >= 0.5, 1, 0), cost
