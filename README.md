# EasyMLP
EasyMLP is just a simple implementation of a Multi-Layer Perceptron written using pure NumPy. As the name suggests, you can create as many layers with as many units as you want. Don't worry, we won't tell ;).

There are a couple of files in this repo that need a bit of explaining:

- `util.py` contains a lot of utility functions that are needed for a neural network to function. Stuff such as cost functions, activation functions, their derivatives, indicator matrix convertors etc.
There is a `load_example_data()` function in this file, that takes in the `train.csv` file from the infamous MNIST dataset and processes it and makes it ready for use in a neural network. We do not provide that dataset here, but you can easily find it if you just google it. 
Keep in mind that we only use the `train.csv` file and **NOT** the `test.csv` file. The reasons remains unknown, even to me, but it's mostly because my system can barely handle pre-proessing and training on the split-up `train.csv` file, but I digress.

- `mlp.py` contains the actual neural network code. This is a very simple implementation of a Multi-Layer Perceptron with mini-batch gradient descent and the option to use *momentum* **OR** *adam* as an optimizer. You can also choose between *sigmoid*, *tanh* or *relu* as your activation function, but that's about it. No, we do not have *leaky relu* here and we never will. Count on it.
The file is well commented/documented and I have tried my best to implement things correctly, but should you have any questions or complaints, feel free to ask them. You can also report any issues that you see and we'll take care of them. Perhaps together!  

- `test.py` contains an example on how you should train your model using our code. It's pretty simple really, if you have used `sklearn`, you can use this.  

Keep in mind that you need to have the MNIST dataset files in the same directory as our files.