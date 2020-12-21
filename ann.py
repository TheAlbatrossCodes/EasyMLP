import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from util import *

class ANN():
    def __init__(self):
        pass
    
    def __generate_weights(self, D, hidden_layer_sizes, K):
        """
        Generates weights and biases. Since we're leaving the number of hidden layers up to the user
        this becomes a bit tricky.
        In order to avoid any unnecessary problems, such as vanishing gradients etc., we'll divide each weight marix
        by the square root of the matrix's row number.

        Args:
            D ([int]): Dimensionality of the dataset (number of features)
            hidden_layer_sizes ([list]): the size & the number of the hidden layer fit() was called with
            K ([int]): Number of classes in your dataset (targets)

        Returns:
            weights, biases [dict]: dictionaries containing the requisite number of weights and biases needed to train the model 
        """
        # find no. hidden layers
        size = len(hidden_layer_sizes)
        weights = {}
        biases = {}
        
        # input to first hidden layer weights
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

    def __forward(self, X, weights, biases, activation_func):
        """
        This is the feed-forward section of the neural network, where we actually feed the network data
        and ask it to predict it's class for us

        Args:
            X (array): your training data
            weights (dict): a dictionary containing the model's weights
            biases (dict): a dict containing the model's biases
            activation_func (str): activation function, as given to us by the fit() method

        Returns:
            Z (dict): contains the activated regression function at each hidden layer.
            pY (array): a matrix showing the probability of each data point beloning to each class
        """
        # use dictionary to store different activation functions instead of many ifs
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


    def __train(self, epochs, Xtrain, Ytrain, weights, biases, lr, reg, activation_func, momentum, beta1, adam, beta2, batch_size):
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
                weights, biases = self.__gradient_descent(w_grad, b_grad, weights, biases, lr, reg, momentum, beta1)
            
            # every 100 epoch, report how the training is going to the user
            if e % 100 == 0:
                _, pY = self.__forward(Xtrain, weights, biases, activation_func)
                ctrain = cost(Ytrain_ind, pY)
                train_cost.append(ctrain)
                pred = prediction(pY)
                e_rate = error_rate(Ytrain, pred) 
                print("Epoch: {}, Train cost: {:.5f}, Error rate: {:.2f}".format(e, ctrain, e_rate))
        
        _, pY = self.__forward(Xtrain, weights, biases, activation_func)
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
         _, pYtest = self.__forward(Xtest, self.weights, self.biases, self.activation_func)
         return prediction(pYtest)

    def score(self, predicted_y, actual_y):
        # returns classifcation rate to the user
        return np.mean(predicted_y == actual_y)


    def fit(self, X, Y, hidden_layer_sizes, lr=1e-6, reg=0.01, epochs=1000, batch_size=512, momentum=False, beta1=0.9, adam=False, beta2=0.99, activation_func='relu'):
        # the main function the user calls to create a model, this will call the necessary functions to train and report to the user
        X, Y = shuffle(X, Y)

        D = X.shape[1]
        K = len(set(Y))
        
        # store activation function in an attribute as well
        self.activation_func = activation_func

        # use self instead of global, because user might want a report of weights/biases
        self.weights, self.biases = self.__generate_weights(D, hidden_layer_sizes, K)
        self.__train(epochs, X, Y, self.weights, self.biases, lr, reg, activation_func, momentum, beta1, adam, beta2, batch_size)


Xtrain, Ytrain, Xtest, Ytest = load_example_data(split=True)

test = ANN()
test.fit(Xtrain, Ytrain, [200, 200], epochs=200, momentum=True)
test.plot_cost()

pred = test.predict(Xtest)
score = test.score(pred, Ytest)
print("Final classification rate: {:.2f}".format(score))
