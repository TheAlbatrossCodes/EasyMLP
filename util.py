"""
Hosts a couple of useful Machine Learning functions
Returns a ready-to-be-used MNIST dataset (you must have the initial file, however)
"""

import numpy as np


# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# tanh
def tanh(z):
    return np.tanh(z)


# relu
def relu(z):
    return z * (z > 0)


# softmax
def softmax(a):
    exp_a = np.exp(a)
    return exp_a / exp_a.sum(axis=1, keepdims=True)


# y2indicator
def y2indicator(y):
    N = len(y)
    D = len(set(y))
    y_indicator = np.zeros((N, D))

    for i in range(N):
        y_indicator[i, y[i]] = 1
    
    return y_indicator


# binary cost
def binary_cost(T, pY):
    return -(T*np.log(pY) + (1-T)*np.log(1-pY)).sum()


# cost
def cost(T, pY_softmax):
    return -(T*np.log(pY_softmax)).mean()


# predict
def predict(pY_softmax):
    return np.argmax(pY_softmax, axis=1)


# score
def classification_rate(T, pY):
    return np.mean(T == pY)


def error_rate(T, pY):
        return np.mean(T != pY)