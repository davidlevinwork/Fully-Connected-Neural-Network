import sys
import numpy as np

# <editor-fold desc="Global Variables">

EPOCHS = 20
TRAIN_RATIO = 0.85
LEARNING_RATE = 0.01
NUM_OF_CLASSES = 10
HIDDEN_LAYER_SIZE = 100

# </editor-fold>


# Function role: shuffle the given data.
def shuffle(X_train, Y_train):
    data = list(zip(X_train, Y_train))
    np.random.shuffle(data)
    X_train, Y_train = zip(*data)
    X_train = np.asarray(X_train)
    Y_train = np.asarray(Y_train)
    return X_train, Y_train


# Function role: scale the given data.
def scaleData(X_train, X_test):
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    return X_train, X_test


# Function role: split the given data to train & validation arrays (train ratio set to 0.8)
def splitData(X_train, Y_train):
    X_train = X_train[:int(TRAIN_RATIO * len(X_train))]
    Y_train = Y_train[:int(TRAIN_RATIO * len(Y_train))]
    X_validation = X_train[int(TRAIN_RATIO * len(X_train)):]
    Y_validation = Y_train[int(TRAIN_RATIO * len(Y_train)):]
    return X_train, Y_train, X_validation, Y_validation


# Function role: write the results to the output file.
def writeData(results):
    with open("test_y", "w") as f:
        f.write("\n".join(map(str, results)))


# Function role: softmax.
def softmax(x):
    exps = np.exp(x - np.max(x))
    return exps / np.sum(exps, axis=0)


# Function role: sigmoid.
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Function role: sigmoid derivative.
def sigmoidDerivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


# Function role: calculate the loss value.
def getLoss(y, prediction):
    return -np.log(prediction[int(y)])



class NeuralNetwork:
    def __init__(self, X_train, Y_train, X_test):
        """
        CTOR.
        Initiation the train & test values.
        Initiation of the Neural Network layers:
            - The first layer (input): 784 X 100. Bias: 100 X 1
            - The second layer (hidden): 100 X 10. Bias: 10 X 1
        """
        self.train_x = X_train
        self.train_y = Y_train
        self.test_x = X_test
        # Weight & Bias of input layer
        self.W_input = np.random.uniform(-0.05, 0.05, (HIDDEN_LAYER_SIZE, X_train.shape[1]))
        self.b_input = np.random.uniform(-0.05, 0.05, (HIDDEN_LAYER_SIZE, 1))
        # Weight & Bias of hidden layer
        self.W_hidden = np.random.uniform(-0.05, 0.05, (NUM_OF_CLASSES, HIDDEN_LAYER_SIZE))
        self.b_hidden = np.random.uniform(-0.05, 0.05, (NUM_OF_CLASSES, 1))

    def test(self):
        """
        Test function: execute the prediction on the given test data (after we trained the Neural Network).
        """
        results = []
        for x_i in self.test_x:
            results.append(neural_network.predict(x_i, 0))
        writeData(results)

    def predict(self, x, y):
        """
        Prediction function: predict sample's class.
        """
        x = x.reshape(x.size, 1)
        results = self.forwardPropagation(x, y)
        return np.argmax(results["CenterLayer"]["output"])

    def forwardPropagation(self, x, y):
        """
        Forward propagation: compute all layers from input to output and put results in a dictionary.
        The computation will start from the first layer (input layer) until the last hidden layer (only 1 layer..).
        """
        z_input_layer = np.dot(self.W_input, x) + self.b_input
        h_input_layer = sigmoid(z_input_layer)
        z_hidden_layer = np.dot(self.W_hidden, h_input_layer) + self.b_hidden
        h_hidden_layer = softmax(z_hidden_layer)
        return {"Function": {"input": x, "output": y}, "FirstLayer": {"input": z_input_layer, "output": h_input_layer},
                "CenterLayer": {"input": z_hidden_layer, "output": h_hidden_layer}}

    def backPropagation(self, params):
        """
        Backward propagation: compute all layers from output to input and put results in a dictionary.
        The computation will start from the hidden layer (only 1 layer..) until the last first layer (input layer).
        """
        y_prob_vec = np.zeros((NUM_OF_CLASSES, 1))
        y_prob_vec[int(params["Function"]["output"])] = 1

        z_hidden_gradient = params["CenterLayer"]["output"] - y_prob_vec
        W_hidden_gradient = np.dot(z_hidden_gradient, params["FirstLayer"]["output"].T)
        # dL/dz2 * dz2/db2
        b_hidden_gradient = np.copy(z_hidden_gradient)

        z_input_gradient = np.dot(self.W_hidden.T, z_hidden_gradient) * sigmoidDerivative(params["FirstLayer"]["input"])
        W_input_gradient = np.dot(z_input_gradient, params["Function"]["input"].T)
        # dL/dz2 * dz2/dh1 * dh1/dz1 * dz1/db1
        b_input_gradient = np.copy(z_input_gradient)
        return {"FirstLayer": {"weight": W_input_gradient, "bias": b_input_gradient},
                "CenterLayer": {"weight": W_hidden_gradient, "bias": b_hidden_gradient}}

    def updateWeights(self, gradients):
        """
        Update weights function: Update the weights during the backward propagation process.
        """
        self.W_input -= LEARNING_RATE * gradients["FirstLayer"]["weight"]
        self.b_input -= LEARNING_RATE * gradients["FirstLayer"]["bias"]
        self.W_hidden -= LEARNING_RATE * gradients["CenterLayer"]["weight"]
        self.b_hidden -= LEARNING_RATE * gradients["CenterLayer"]["bias"]

    def train(self):
        """
        Train function: execute the learning process of the neural network.
        """
        # Split the data to train & validation sets
        training_x, training_y, validation_x, validation_y = splitData(self.train_x, self.train_y)

        for epoch in range(EPOCHS):
            loss_sum = 0
            # Shuffle the data in each iteration
            training_x, training_y = shuffle(training_x, training_y)


            for x_i, y_i in zip(training_x, training_y):
                # Reshape the data size
                x_i = x_i.reshape(x_i.size, 1)
                # Forward propagation & save the results
                fPropagation_cache = self.forwardPropagation(x_i, y_i)

                # Calculate the loss value
                loss = getLoss(fPropagation_cache["Function"]["output"], fPropagation_cache["CenterLayer"]["output"])
                loss_sum += loss

                # Backward propagation & save the results
                bPropagation_cache = self.backPropagation(fPropagation_cache)
                # Update the values after the backward propagation process
                self.updateWeights(bPropagation_cache)


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Error: Insufficient arguments!")
        sys.exit()
    else:
        train_x, train_y, test_x = np.loadtxt(sys.argv[1]), np.loadtxt(sys.argv[2]), np.loadtxt(sys.argv[3])

        # Shuffle the data
        train_x, train_y = shuffle(train_x, train_y)

        # Scale the data
        train_x, test_x = scaleData(train_x, test_x)

        # Create & train the neural network
        neural_network = NeuralNetwork(train_x, train_y, test_x)
        neural_network.train()
        # Test the neural network
        neural_network.test()
