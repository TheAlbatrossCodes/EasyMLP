"""
Hosts a couple of useful Machine Learning functions
Returns a ready-to-be-used MNIST dataset (you must have the initial file, however)
"""

import numpy as np
import csv


# sigmoid
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# derivation of sigmoid
def deriv_sigmoid(x):
    return x * (1 - x)


# wrapper around numpy's tanh
def tanh(z):
    return np.tanh(z)


# derivative of tanh
def deriv_tanh(x):
    return 1 - x**2


# relu
def relu(z):
    return z * (z > 0)


# derivative of relu
def deriv_relu(x):
    return x * (x > 0)


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
def prediction(pY_softmax):
    return np.argmax(pY_softmax, axis=1)


# score
def classification_rate(T, pY):
    return np.mean(T == pY)


def error_rate(T, pY):
        return np.mean(T != pY)


def load_example_data(split=False, split_by=0.33):
    # read MNIST data
    with open("train.csv", 'r') as f:
        csv_file = csv.reader(f, delimiter=',')
        next(csv_file)
        content = []
        for row in csv_file:
            content.append(row)

    # turn data into numpy array and shuffle
    data = np.array(content, dtype=np.float32)
    np.random.shuffle(data)
    Y = data[:, 0].astype(np.int32)
    X = data[:, 1:]

    # normalize X
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    np.place(X_std, X_std==0, 1)

    X = (X - X_mean) / X_std

    if split:
        N = X.shape[0]
        split_amount = int(N*split_by)
        Xtrain, Ytrain = X[:-split_amount], Y[:-split_amount]
        Xtest, Ytest = X[-split_amount:], Y[-split_amount:]
        return Xtrain, Ytrain, Xtest, Ytest

    else:
        return X, Y
