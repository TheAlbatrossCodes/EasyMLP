import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import *

class ANN():
    def __init__(self):
        pass
    
    def __generate_weights(self, D, hidden_layer_sizes, K):
        # find no. hidden layers
        size = len(hidden_layer_sizes)
        weights = {}
        biases = {}
        
        # input to first hidden later weights
        weights[0] = np.random.randn(D, hidden_layer_sizes[0]) / np.sqrt(D)
        biases[0] = np.zeros(hidden_layer_sizes[0])
        
        # the weights beween hidden layers
        for i in range(size - 1):
            weights[i+1] = np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i+1]) / np.sqrt(hidden_layer_sizes[i])
            biases[i+1] = np.zeros(hidden_layer_sizes[i+1])

        # last hidden layer to output layer weight
        weights[size] = np.random.randn(hidden_layer_sizes[-1], K) / np.sqrt(hidden_layer_sizes[-1])
        biases[size] = np.zeros(K)
        
        return weights, biases

    def __forward(self, X, weights, biases, activation_func='relu'):
        # use dictionary to store different activation function instead of many ifs
        activate = {'relu':relu, 'tanh': tanh, 'sigmoid':sigmoid}

        layers = len(weights.keys())
        Z = {}     

        # Z at first hidden layer
        Z[0] = activate[activation_func](X.dot(weights[0]) + biases[0])

        # Z at other hidden layers
        for i in range(layers - 2):
            z_list = list(Z.keys())
            Z[i+1] = activate[activation_func](Z[z_list[-1]].dot(weights[i+1]) + biases[i+1])
        
        # pY_given_x
        z_list = list(Z.keys())
        pY = softmax(Z[z_list[-1]].dot(weights[layers-1]) + biases[layers-1])

        return Z, pY
    
    def __gradients(self, X, Y, weights, biases, Z, pY, activation_func):
        # store the derivative of each activation function in dict.
        activate_deriv = {'relu':deriv_relu, 'tanh': deriv_tanh, 'sigmoid':deriv_sigmoid}
        
        n_grad = len(weights.keys())
        n_z = len(Z.keys())
        
        w_grad = {}
        b_grad = {}
        dZ = {}

        # gradient of weights between output layer and last hidden layer
        w_grad[n_grad-1] = Z[n_z-1].T.dot(pY - Y)
        b_grad[n_grad-1] = (pY - Y).sum(axis=0)
        
        # find dZ for all hidden layers
        dZ[n_z - 1] = (pY - Y).dot(weights[n_z].T) * activate_deriv[activation_func](Z[n_z - 1])

        for i in reversed(range(n_grad - 2)):
            dZ[i] = dZ[i+1].dot(weights[i + 1].T) * activate_deriv[activation_func](Z[i])

        # use dZ to calculate gradients
        n_dZ = len(dZ.keys())
        for j in reversed(range(n_dZ)):
            if j != 0:
                w_grad[j] = Z[j-1].T.dot(dZ[j])
                b_grad[j] = dZ[j].sum(axis=0)
            else:
                w_grad[j] = X.T.dot(dZ[j])
                b_grad[j] = dZ[j].sum(axis=0)

        return w_grad, b_grad    

    def __gradient_descent(self, w_grad, b_grad, weights, biases, lr, reg, momentum, beta):
        # update every weight vector / momentum

        n_weights = len(weights.keys())
        for i in range(n_weights):
            if momentum:
                global Vw, Vb
                Vw[i] = beta * Vw[i] - lr * (w_grad[i] + reg*weights[i])
                Vb[i] = beta * Vb[i] - lr * (b_grad[i] + reg*biases[i])
                weights[i] = weights[i] + Vw[i]
                biases[i] = biases[i] + Vb[i]
            else:
                weights[i] = weights[i] - lr * (w_grad[i] + reg*weights[i])
                biases[i] = biases[i] - lr * (b_grad[i] + reg*biases[i])
        
        return weights, biases


    def __train(self, epochs, Xtrain, Ytrain, weights, biases, lr, reg, activation_func, momentum, beta, show_fig, batch_size):
        # handle most of the learning by calling relevant functions and looping
        Ytrain_ind = y2indicator(Ytrain)
        N = Xtrain.shape[0]
        no_batch = N // batch_size

        # global so we can plot it later
        global train_cost
        train_cost = []
        
        if momentum:
            global Vw, Vb
            Vw = {key: 0 for key in weights.keys()}
            Vb = {key: 0 for key in biases.keys()}

        for e in range(epochs):
            tmp_x, tmp_y = shuffle(Xtrain, Ytrain_ind)
            
            for i in range(no_batch):
                X = tmp_x[no_batch*batch_size:no_batch*batch_size+batch_size]
                Y = tmp_y[no_batch*batch_size:no_batch*batch_size+batch_size]
                
                Z, pY = self.__forward(X, weights, biases, activation_func)

                w_grad, b_grad = self.__gradients(X, Y, weights, biases, Z, pY, activation_func)
                weights, biases = self.__gradient_descent(w_grad, b_grad, weights, biases, lr, reg, momentum, beta)
            
            # every 100 epoch, report how the training is going to the user
            if e % 100 == 0:
                _, pY = self.__forward(Xtrain, weights, biases)
                ctrain = cost(Ytrain_ind, pY)
                train_cost.append(ctrain)
                pred = prediction(pY)
                e_rate = error_rate(Ytrain, pred) 
                print("Epoch: {}, Train cost: {:.5f}, Error rate: {:.2f}".format(e, ctrain, e_rate))
        
        _, pY = self.__forward(Xtrain, weights, biases)
        ctrain = cost(Ytrain_ind, pY)
        pred = prediction(pY)
        e_rate = error_rate(Ytrain, pred) 
        print("\nFinal training cost: {:.5f}, Final training error rate: {:.2f}".format(ctrain, e_rate))

    def plot_cost(self):
        # plot how the cost has changed during epochs
        global train_cost
        plt.plot(train_cost)
        plt.show(block=False)

    def predict(self, Xtest):
        # use trained weights to predict Y for test set
         _, pYtest = self.__forward(Xtest, self.weights, self.biases)
         return prediction(pYtest)

    def score(self, predicted_y, actual_y):
        # returns classifcation rate to the user
        return np.mean(predicted_y == actual_y)


    def fit(self, X, Y, hidden_layer_sizes, lr=1e-6, reg=0.01, epochs=1000, batch_size=512, momentum=False, beta=0.9, activation_func='relu', show_fig=True):
        # the main function the user calls to create a model, this will call the necessary functions to train and report to the user
        X, Y = shuffle(X, Y)

        D = X.shape[1]
        K = len(set(Y))
        
        # use self instead of global, because user might want a report of weights/biases
        self.weights, self.biases = self.__generate_weights(D, hidden_layer_sizes, K)
        self.__train(epochs, X, Y, self.weights, self.biases, lr, reg, activation_func, momentum, beta, show_fig, batch_size)


Xtrain, Ytrain, Xtest, Ytest = load_example_data(split=True)

test = ANN()
test.fit(Xtrain, Ytrain, [200, 200], epochs=200, momentum=True)
test.plot_cost()

pred = test.predict(Xtest)
score = test.score(pred, Ytest)
print("Final classification rate: {:.2f}".format(score))
