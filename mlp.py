"""
This is one of the very first programs I wrote. In this program, we try
to use the recursive nature of the Backpropagation algorithm to
implement a fully customizable MLP model.
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import util

class MLPClassifier:    
    def __generate_weights(self, D, hidden_layer_sizes, K):
        """
        Generates weights and biases. Since we're leaving the number of
        hidden layers up to the user this becomes a bit tricky.
        In order to avoid any unnecessary problems, such as vanishing
        gradients etc., we'll divide each weight marix by the square
        root of the matrix's row number.l 
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
        This is the feed-forward section of the neural network, where we
        actually feed the network data and ask it to predict it's class
        for us
        """
        # use dictionary to store different activation functions instead of many ifs
        activate = {'relu':util.relu, 'tanh':util.tanh, 'sigmoid':util.sigmoid}

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
        pY = util.softmax(Z[z_list[-1]].dot(weights[layers-1]) + biases[layers-1])

        return Z, pY
    
    def __gradients(self, X, Y, weights, biases, Z, pY, activation_func):
        """
        This function will calculate the gradients of weights and biases
        We don't have automatic derivations, so we'll have to calulate
        them by hand and use the recursive nature of backprop to make
        things work in many layers.
        """
        # store the derivative of each activation function in dict.
        activate_deriv = {'relu':util.deriv_relu, 'tanh':util.deriv_tanh, 'sigmoid':util.deriv_sigmoid}
        
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

    def __gradient_descent(self, w_grad, b_grad, weights, biases, lr, reg, momentum, beta1, adam, beta2, epsilon):
        """
        This function does the gradient descent. Keep in mind that we
        can only have one optimizer. This means if we can either have
        momentum or Adam, and if both are true, Adam takes precedence.
        """
        n_weights = len(weights.keys())
        for i in range(n_weights):
            if adam:
                global mw, mb, vw, vb, mw_hat, mb_hat, vw_hat, vb_hat, t
                mw[i] = beta1*mw[i] + (1-beta1)*w_grad[i] + reg*weights[i]
                mb[i] = beta1*mb[i] + (1-beta1)*b_grad[i] + reg*biases[i]
                vw[i] = beta2*vw[i] + (1-beta2)*(w_grad[i] + reg*weights[i])**2
                vb[i] = beta2*vb[i] + (1-beta2)*(b_grad[i] + reg*biases[i])**2

                mw_hat[i] = mw[i] / (1-beta1**t)
                mb_hat[i] = mb[i] / (1-beta1**t)
                vw_hat[i] = vw[i] / (1-beta2**t)
                vb_hat[i] = vb[i] / (1-beta2**t)
                
                weights[i] = weights[i] - lr*(mw_hat[i]/np.sqrt(vw_hat[i] + epsilon))
                biases[i] = biases[i] - lr*(mb_hat[i]/np.sqrt(vb_hat[i] + epsilon))

            elif momentum:
                global Vw, Vb
                Vw[i] = beta1 * Vw[i] - lr * (w_grad[i] + reg*weights[i])
                Vb[i] = beta1 * Vb[i] - lr * (b_grad[i] + reg*biases[i])
                weights[i] = weights[i] + Vw[i]
                biases[i] = biases[i] + Vb[i]

            else:
                weights[i] = weights[i] - lr * (w_grad[i] + reg*weights[i])
                biases[i] = biases[i] - lr * (b_grad[i] + reg*biases[i])
        
        return weights, biases


    def __train(self, epochs, Xtrain, Ytrain, weights, biases, lr, reg, activation_func, momentum, beta1, adam, beta2, epsilon, batch_size):
        """
        This is where training gets done, meaning this is where we do
        our epochs, calculate cost, handle batches/momentum etc.
        """
        Ytrain_ind = util.y2indicator(Ytrain)
        N = Xtrain.shape[0]
        no_batch = N // batch_size

        # global so we can plot it later
        global train_cost
        train_cost = []
        
        if momentum:
            global Vw, Vb
            Vw = {key: 0 for key in weights.keys()}
            Vb = {key: 0 for key in biases.keys()}
        if adam:
            global mw, mb, vw, vb, mw_hat, mb_hat, vw_hat, vb_hat, t
            t=1
            mw, mw_hat = {key: 0 for key in weights.keys()}, {key: 0 for key in weights.keys()}
            mb, mb_hat = {key: 0 for key in biases.keys()}, {key: 0 for key in biases.keys()}
            vw, vw_hat = {key: 0 for key in weights.keys()}, {key: 0 for key in weights.keys()}
            vb, vb_hat = {key: 0 for key in biases.keys()}, {key: 0 for key in weights.keys()}

        for e in range(epochs):
            tmp_x, tmp_y = shuffle(Xtrain, Ytrain_ind)
            
            for i in range(no_batch):
                X = tmp_x[no_batch*batch_size:no_batch*batch_size+batch_size]
                Y = tmp_y[no_batch*batch_size:no_batch*batch_size+batch_size]
                
                Z, pY = self.__forward(X, weights, biases, activation_func)

                w_grad, b_grad = self.__gradients(X, Y, weights, biases, Z, pY, activation_func)
                weights, biases = self.__gradient_descent(w_grad, b_grad, weights, biases, lr, reg, momentum, beta1, adam, beta2, epsilon)
                if adam:
                    t+=1

            
            # every 100 epoch, report how the training is going to the user
            if e % 100 == 0:
                _, pY = self.__forward(Xtrain, weights, biases, activation_func)
                ctrain = util.cost(Ytrain_ind, pY)
                train_cost.append(ctrain)
                pred = util.prediction(pY)
                e_rate = util.error_rate(Ytrain, pred) 
                print("Epoch: {}, Train cost: {:.5f}, Error rate: {:.2f}".format(e, ctrain, e_rate))
        
        _, pY = self.__forward(Xtrain, weights, biases, activation_func)
        ctrain = util.cost(Ytrain_ind, pY)
        pred = util.prediction(pY)
        e_rate = util.error_rate(Ytrain, pred) 
        print("\nFinal training cost: {:.5f}, Final training error rate: {:.2f}".format(ctrain, e_rate))

    def plot_cost(self):
        """
        You can call this function to plot the training cost of your
        model, but you're better off just doing it yourself
        """
        # plot how the cost has changed during epochs
        global train_cost
        plt.plot(train_cost)
        plt.show(block=False)

    def predict(self, Xtest):
        """
        This is the function through which you test the accuracy of your
        model.
        It bascially takes your test set in and predicts the labels,
        based on the weights you have trained.
        """
        # use trained weights to predict Y for test set
        _, pYtest = self.__forward(Xtest, self.weights, self.biases, self.activation_func)
        return util.prediction(pYtest)

    def score(self, predicted_y, actual_y):
        """
        This function will return the classification rate of the model
        to the user. 
        """
        # returns classifcation rate to the user
        return np.mean(predicted_y == actual_y)


    def fit(self, X, Y, hidden_layer_sizes, lr=1e-6, reg=0.01, epochs=1000, batch_size=512, momentum=False, beta1=0.9, adam=False, beta2=0.99, epsilon=1e-8, activation_func='relu'):
        """
        Creates a neural network based on your specifications.
        Args:
            X (array): the input data
            Y (array): the labels for the input data
            hidden_layer_sizes (list): the number of neurons in each
              hidden layer, must be a list, e.g. [100] will create 1 
              hidden layer with 100 neurons, [100, 100] will create 2 
            hidden layers with 100 neurons each, and so on
            lr (float): The learning rate. Defaults to 1e-6.
            reg (float): Regularization parameter. Defaults to 0.01.
            epochs (int): The number of epochs we train the model for.
              Defaults to 1000.
            batch_size (int, optional): The size of each batch for
              mini-batch gradient descent. Defaults to 512.
            momentum (bool): Whether or not you'd like the mode to be
              trained with momentum. Defaults to False.
            beta1 (float): Sets the momentum parameter. Defaults to 0.9.
            adam (bool): Whether or not you'd like the model to be
              trained with Adam optimizer. If adam and momentum are both
              True, the model will be trained with adam only.
              Defaults to False.
            beta2 (float): Decay parameter (adam parameter). 
              Defaults to 0.99.
            epsilon (float): The adam constant. Defaults to 1e-8.
            activation_func (str): The activation function for the model.
              Can be a string of 'sigmoid', 'tanh' or 'relu'.Defaults to 'relu'.
        """
        # the main function the user calls to create a model, this will
        #  call the necessary functions to train and report to the user
        X, Y = shuffle(X, Y)

        D = X.shape[1]
        K = len(set(Y))

        # take care of accidental int type assignment
        if type(hidden_layer_sizes) == int:
            hidden_layer_sizes = [hidden_layer_sizes]

        # store activation function in an attribute as well
        self.activation_func = activation_func

        # use self instead of global, because user might want a report of weights/biases
        self.weights, self.biases = self.__generate_weights(D, hidden_layer_sizes, K)
        self.__train(epochs, X, Y, self.weights, self.biases, lr, reg, activation_func, momentum, beta1, adam, beta2, epsilon, batch_size)
