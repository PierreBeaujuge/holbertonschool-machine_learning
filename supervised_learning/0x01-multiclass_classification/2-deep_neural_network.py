#!/usr/bin/env python3
"""
Binary Classification
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    define the DeepNeuralNetwork class
    """

    def __init__(self, nx, layers):
        """initialize variables and methods"""
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or not len(layers):
            raise TypeError('layers must be a list of positive integers')
        self.nx = nx
        self.layers = layers
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError('layers must be a list of positive integers')
            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])
            self.__weights['b' + str(i + 1)] = np.zeros(
                layers[i]).reshape(layers[i], 1)

    @property
    def L(self):
        """getter for L"""
        return self.__L

    @property
    def cache(self):
        """getter for cache"""
        return self.__cache

    @property
    def weights(self):
        """getter for weights"""
        return self.__weights

    def forward_prop(self, X):
        """forward propagation function"""
        self.__cache['A0'] = X
        for i in range(self.L):
            Zi = np.matmul(
                self.weights['W' + str(i + 1)], self.cache['A' + str(i)]
            ) + self.weights['b' + str(i + 1)]
            self.__cache['A' + str(i + 1)] = self.sigmoid(Zi)
        return self.cache['A' + str(i + 1)], self.cache

    def sigmoid(self, Y):
        """define the sigmoid activation function"""
        return 1 / (1 + np.exp(-1 * Y))

    def cost(self, Y, A):
        """define the cost function"""
        m = Y.shape[1]
        return (-1 / m) * np.sum(
            Y * np.log(A) + (1 - Y) * (np.log(1.0000001 - A)))

    def evaluate(self, X, Y):
        """function that evaluates the dnn's predictions"""
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        return np.where(A >= 0.5, 1, 0), cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """function that calculates one pass of gradient descent"""
        weights = self.weights.copy()
        for i in range(self.L, 0, -1):
            m = Y.shape[1]
            if i != self.L:
                dZi = np.multiply(np.matmul(
                    weights['W' + str(i + 1)].T, dZi
                ), (self.cache['A' + str(i)] * (1 - self.cache['A' + str(i)])))
            else:
                dZi = self.cache['A' + str(i)] - Y
            dWi = (1 / m) * np.matmul(dZi, self.cache['A' + str(i - 1)].T)
            dbi = (1 / m) * np.sum(dZi, axis=1, keepdims=True)
            self.__weights['W' + str(i)] = weights['W' + str(i)] - alpha * dWi
            self.__weights['b' + str(i)] = weights['b' + str(i)] - alpha * dbi

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """function that trains the dnn"""
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
            A, cache = self.forward_prop(X)
            # backpropagate except for last iteration (5000):
            if i != iterations:
                self.gradient_descent(Y, cache, alpha)
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

    def save(self, filename):
        """function that saves a dnn instance to a file in pkl format"""
        if ".pkl" not in filename:
            filename += ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """function that loads a pkled dnn instance from a file"""
        try:
            with open(filename, 'rb') as f:
                dnn = pickle.load(f)
            return dnn
        except FileNotFoundError:
            return None
