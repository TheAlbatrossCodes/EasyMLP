import numpy as np
import matplotlib as plt
from sklearn.utils import shuffle
from util import *

class ANN():
    def __init__(self):
        pass
    
    def __generate_weights(self, D, hidden_layer_sizes, K):
        size = len(hidden_layer_sizes)
        weights = {}
        biases = {}
        weights[0] = np.random.randn(D, hidden_layer_sizes[0]) / np.sqrt(D)
        biases[0] = np.zeros(hidden_layer_sizes[0])
        for i in range(size - 1):
            weights[i+1] = np.random.randn(hidden_layer_sizes[i], hidden_layer_sizes[i+1]) / np.sqrt(hidden_layer_sizes[i])
            biases[i+1] = np.zeros(hidden_layer_sizes[i+1])

        weights[size] = np.random.randn(hidden_layer_sizes[-1], K) / np.sqrt(hidden_layer_sizes[-1])
        biases[size] = np.zeros(K)
        
        return weights, biases

    def __forward(self, X, weights, biases, activation_func='relu'):
        activate = {'relu':relu, 'tanh': tanh, 'sigmoid':sigmoid}
        layers = len(weights.keys())
        Z = {}
        Z[0] = activate[activation_func](X.dot(weights[0]) + biases[0])

        for i in range(layers - 2):
            z_list = list(Z.keys())
            Z[i+1] = activate[activation_func](Z[z_list[-1]].dot(weights[i+1]) + biases[i+1])
            
        z_list = list(Z.keys())
        pY = softmax(Z[z_list[-1]].dot(weights[layers-1]) + biases[layers-1])

        return Z, pY
    
    def __gradients(self, X, Y, weights, biases, Z, pY, activation_func):
        activate_deriv = {'relu':deriv_relu, 'tanh': deriv_tanh, 'sigmoid':deriv_sigmoid}
        
        n_grad = len(weights.keys())
        n_z = len(Z.keys())
        
        w_grad = {}
        b_grad = {}
        dZ = {}

        w_grad[n_grad-1] = Z[n_z-1].T.dot(pY - Y)
        b_grad[n_grad-1] = (pY - Y).sum(axis=0)
        
        dZ[n_z - 1] = (pY - Y).dot(weights[n_z].T) * activate_deriv[activation_func](Z[n_z - 1])

        for i in reversed(range(n_grad - 2)):
            dZ[i] = dZ[i+1].dot(weights[i + 1].T) * activate_deriv[activation_func](Z[i])
        
        n_dZ = len(dZ.keys())
        for j in reversed(range(n_dZ)):
            if j != 0:
                w_grad[j] = Z[j-1].T.dot(dZ[j])
                b_grad[j] = dZ[j].sum(axis=0)
            else:
                w_grad[j] = X.T.dot(dZ[j])
                b_grad[j] = dZ[j].sum(axis=0)

        return w_grad, b_grad    

    def __gradient_descent(self, w_grad, b_grad, weights, biases, lr, reg, momentum, beta=0.9):
        n_weights = len(weights.keys())
        for i in range(n_weights):
            if momentum:
                self.Vw[i] = beta * self.Vw[i] - lr * (w_grad[i] + reg*weights[i])
                self.Vb[i] = beta * self.Vb[i] - lr * (b_grad[i] + reg*biases[i])
                weights[i] = weights[i] + self.Vw[i]
                biases[i] = biases[i] + self.Vb[i]
            else:
                weights[i] = weights[i] - lr * (w_grad[i] + reg*weights[i])
                biases[i] = biases[i] - lr * (b_grad[i] + reg*biases[i])
        
        return weights, biases


    def __train(self, epochs, Xtrain, Ytrain, weights, biases, lr, reg, activation_func, momentum, batch_size=512):
        Ytrain_ind = y2indicator(Ytrain)
        N = Xtrain.shape[0]
        no_batch = N // batch_size
        train_cost = []
        if momentum:
            self.Vw = {key: 0 for key in weights.keys()}
            self.Vb = {key: 0 for key in biases.keys()}

        for e in range(epochs):
            tmp_x, tmp_y = shuffle(Xtrain, Ytrain_ind)
            
            for i in range(no_batch):
                X = tmp_x[no_batch*batch_size:no_batch*batch_size+batch_size]
                Y = tmp_y[no_batch*batch_size:no_batch*batch_size+batch_size]
                
                Z, pY = self.__forward(X, weights, biases, activation_func)

                w_grad, b_grad = self.__gradients(X, Y, weights, biases, Z, pY, activation_func)
                weights, biases = self.__gradient_descent(w_grad, b_grad, weights, biases, lr, reg, momentum=momentum)
            
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
        print("\nFinal cost: {:.5f}, Final error rate: {:.2f}".format(ctrain, e_rate))
        

    def predict(self, Xtest):
         _, pYtest = self.__forward(Xtest, self.weights, self.biases)
         return prediction(pYtest)


    def score(self, predicted_y, actual_y):
        return np.mean(predicted_y == actual_y)


    def fit(self, X, Y, hidden_layer_sizes, lr=1e-6, reg=0.01, epochs=1000, momentum=False, activation_func='relu', show_fig=True):
        X, Y = shuffle(X, Y)

        D = X.shape[1]
        K = len(set(Y))
        self.weights, self.biases = self.__generate_weights(D, hidden_layer_sizes, K)
        self.__train(epochs, X, Y, self.weights, self.biases, lr, reg, momentum=momentum, activation_func=activation_func)


        

Xtrain, Ytrain, Xtest, Ytest = load_example_data(split=True)

test = ANN()
test.fit(Xtrain, Ytrain, [200], epochs=500, momentum=False)

pred = test.predict(Xtest)
score = test.score(pred, Ytest)
print("Final classification rate: {:.2f}".format(score))
