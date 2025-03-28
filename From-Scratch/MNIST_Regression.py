import numpy as np
import pandas as pd
import matplotlib as plt

class MNIST_Regression:

    layer_0_size = 0                           # input layer has 784 neurons (one for each pixel)
    layer_1_size = 16                          # hidden layer 1 has 16 neurons
    layer_2_size = 16                          # hidden layer 2 has 16 neurons
    layer_3_size = 10                          # output layer has 10 neurons (one for each predicted digit 1 - 9)

    learning_ratio = 0.01                      # ratio for the rate at which the model learns

    def __init__(self):

        X_validation, Y_validation, X_training, Y_training = self.load_data()

        MNIST_Regression.layer_0_size = X_validation.shape[0]

        # TRAINING
        self.gradient_descent(X_training, Y_training, 500)

        # VALIDATION
        # self.gradient_descent(X_validation, Y_validation, 500)



    def gradient_descent(self, X, Y, epochs):
        m = X.shape[1] 
        activations, weights, biases, weighted_sums = self.init_neural_net(m)
        weight_nudges, bias_nudges, weighted_sum_changes = self.init_nudges(m)

        for i in range(epochs):
            activations, weights, biases, weighted_sums = self.forward_propogration(X, m, activations, weights, biases, weighted_sums)
            weighted_sum_changes, weight_nudges, bias_nudges = self.back_propogration(Y, m, activations, weights, weighted_sum_changes, weight_nudges, bias_nudges)



    def init_neural_net (self, m):    
        # Create matrices for each layer of the Neural Network
        A_0 = np.empty(MNIST_Regression.layer_0_size, m)
        A_1 = np.empty(MNIST_Regression.layer_1_size, m)
        A_2 = np.empty(MNIST_Regression.layer_2_size, m)
        A_3 = np.empty(MNIST_Regression.layer_3_size, m)

        # List of all the activation layers
        activations = [A_0, A_1, A_2, A_3]

        # Create matrices for set of weights between layers in the Neural Network
        W_1 = np.random.uniform(-0.5, 0.5, size = (MNIST_Regression.layer_1_size, MNIST_Regression.layer_0_size))
        W_2 = np.random.uniform(-0.5, 0.5, size = (MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size))
        W_3 = np.random.uniform(-0.5, 0.5, size = (MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size))

        # List of all the weights
        weights = [W_1, W_2, W_3]
    
        # Create matrices for set of biases between layers in the Neural Network
        B_1 = np.zeros(MNIST_Regression.layer_1_size, m)
        B_2 = np.zeros(MNIST_Regression.layer_2_size, m)
        B_3 = np.zeros(MNIST_Regression.layer_3_size, m)

        # List of all the biases
        biases = [B_1, B_2, B_3]

        # Create matrices for weighter sum between layers in the Neural Network
        Z_1 = np.empty(MNIST_Regression.layer_1_size, m)
        Z_2 = np.empty(MNIST_Regression.layer_2_size, m)
        Z_3 = np.empty(MNIST_Regression.layer_3_size, m)

        # List of all the weighted sums
        weighted_sums = [Z_1, Z_2, Z_3]

        return activations, weights, biases, weighted_sums



    def init_nudges(self, m):
        # Create matrices that contain nudges to add to weights between layers
        dW_1 = np.empty(MNIST_Regression.layer_1_size, MNIST_Regression.layer_2_size)
        dW_2 = np.empty(MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size)
        dW_3 = np.empty(MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size)

        # List of all the weight nudge matrices
        weight_nudges = [dW_1, dW_2, dW_3]

        # Create matrices that contain nudges to add to biases between layers
        dB_1 = np.empty(MNIST_Regression.layer_1_size, 1)
        dB_2 = np.empty(MNIST_Regression.layer_2_size, 1)
        dB_3 = np.empty(MNIST_Regression.layer_3_size, 1)

        # List of all the bias nudge matrices
        bias_nudges = [dB_1, dB_2, dB_3]

        # Create matrices that contain the change in the weighted sums for dW and dB computation
        dZ_1 = np.empty(MNIST_Regression.layer_1_size, m)
        dZ_2 = np.empty(MNIST_Regression.layer_2_size, m)
        dZ_3 = np.empty(MNIST_Regression.layer_3_size, m)

        # Create list of all the changes in the weighted sums
        weighted_sum_changes = [dZ_1, dZ_2, dZ_3]

        return weight_nudges, bias_nudges, weighted_sum_changes



    def load_data(self):
        datafile = pd.read_csv('MNIST_Digits.zip')   # load in training data (each training sample is a row that is 785 columns long (row 1 = expected digit, row 2 -> 785 is eac pixel))
        data = np.array(datafile)
        np.random.shuffle(data)
        self.samples, self.dim_pixels = data.shape    # TODO: DIM PIXEL ISSUE???

        # set up validation data to be samples 0 - 1000
        validation_data = data[0 : 1000].T
        Y_validation = validation_data[0]
        X_validation = validation_data[1 : self.dim_pixels]

        training_data = data[1000 : self.samples].T
        Y_training = training_data[0]
        X_training = training_data[1 : self.dim_pixels]
        
        return X_validation, Y_validation, X_training, Y_training



    # TODO: CREATE FUNCTION THAT CAN TURN THE DATA IN Y TO 0'S & 1'S



    # Forward progpagation calculates all the weighted sums for each activation layer
    def forward_propogration(self, X, activation_layers, weights, weighted_sums, biases):
        activation_layers[0] = X
        for i in range(len(activation_layers)):
            weighted_sums[i] = weights[i].dot(activation_layers[i]) + biases[i]
            activation_layers[i + 1] = self.sigmoid(weighted_sums[i])    
        
        return activation_layers, weights, biases, weighted_sums



    # Sigmoid function squishes all the weighted sums between 0 - 1 for compatibility
    # with neuron activation layers
    def sigmoid(self, Z):
        return 1/(1 + np.exp(-self.Z))



    def back_propogration(self, Y, samples, activation_layers, weights, weighted_sum_changes, weight_nudges, bias_nudges):
        # TODO PUT THE ONE HOT FUNCTION CALL HERE TO MODIFY Y
        i = len(weighted_sum_changes) - 1
        while i >= 0:
            if i == len(weighted_sum_changes) - 1:
                weighted_sum_changes[i] = activation_layers[len(activation_layers) - 1] - Y
            else:
                weighted_sum_changes[i] = np.dot(weights[i + 1].T, weighted_sum_changes[i + 1], )# WHAT IS G PRIME OF Z??

            weight_nudges[i] = (1 / samples) * np.dot(weighted_sum_changes[i], activation_layers[i].T)
            bias_nudges[i]= (1 / samples) * np.sum(weighted_sum_changes, axis = 1)

            i -= 1

        return weighted_sum_changes, weight_nudges, bias_nudges



    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (self.learning_ratio * self.weight_nudges[i])
            self.biases[i] = self.biases[i] - (self.learning_ratio * self.bias_nudges[i])
        return

    

