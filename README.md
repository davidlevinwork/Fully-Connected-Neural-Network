# Fully Connected Neural Network
MLP (fully connected neural network) from scratch using numpy on the MNIST dataset (Digit recognition).

1. [General](#General)
    - [Background](#background)
    - [Running Instructions](https://github.com/tomershay100/Fully-Connected-Neural-Network/blob/main/README.md#running-instructions)
2. [Dependencies](#dependencies) 

## General

### Background
Implementation of a neural network for digit classification, on the MNIST dataset, which takes as an input a ``28*28`` grayscale image (``784`` floating point values of pixels between ``0-255``).

### Net Structure
The network has one hidden layer in size 128 (default) and it performs multiple epochs (20 by default) and trains the model by minimizing the Negative Log Likelihood (NLL) with the ``Sigmoid`` and ``Softmax`` activation functions.

During learning, the network verifies its accuracy on an independent set of data (about ``10%`` of the training set) on which learning is not performed. This group is called a ``validation set``. After all the epochs, the network saves its best condition, the weights that resulted the maximum accuracy on the validation set, to prevent overfitting.

Finally, the network exports a graph of the accuracy on the validation and the training sets, by the number of epochs, and verifies the accuracy on the testing set.

### Running Instructions

The program gets several arguments:
* flag ```-train_x TRAIN_X_PATH``` for the training images file path (file that contains 784 values in each row).
* flag ```-train_y TRAIN_Y_PATH``` for the training labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).
* flag ```-test_x TEST_X_PATH``` for the testing images file path (file that contains 784 values in each row).
* flag ```-test_y TEST_Y_PATH``` for the testing labels file path (file that contains one value between ``0-9`` in each row and has the same rows number as the train_x file).


Note that for using the dataset given in this repo, you need to unzip the dataset.zip folder.
## Dependencies
* [Python 3.6+](https://www.python.org/downloads/)
* [NumPy](https://numpy.org/install/)
* [Matplotlib](https://matplotlib.org/stable/users/installing.html)
