import numpy as np
import pandas as pd
import matplotlib as plt
# np.set_printoptions(threshold=np.inf)
class MNIST_Regression:

    layer_0_size = 0                           # input layer has 784 neurons (one for each pixel)
    layer_1_size = 16                          # hidden layer 1 has 16 neurons
    layer_2_size = 16                          # hidden layer 2 has 16 neurons
    layer_3_size = 10                          # output layer has 10 neurons (one for each predicted digit 1 - 9)

    learning_ratio = 0.1                    # ratio for the rate at which the model learns

    datafile_name = 'train.csv.zip'           # name of the input training and validation data file

    num_epochs = 500

    def __init__(self):

        X_validation, Y_validation, X_training, Y_training = self.load_data()

        MNIST_Regression.layer_0_size = X_validation.shape[0]

        # TRAINING
        self.gradient_descent(X_training, Y_training, MNIST_Regression.num_epochs)

        # VALIDATION
        # self.gradient_descent(X_validation, Y_validation, 500)



    def gradient_descent(self, X, Y, epochs):
        m = X.shape[1] 
        activations, weights, biases, weighted_sums = self.init_neural_net(m)
        weight_nudges, bias_nudges, weighted_sum_changes = self.init_nudges(m)

        for i in range(epochs):
            activations, weights, biases, weighted_sums = self.forward_propogration(X, activations, weights, weighted_sums, biases)
            weighted_sum_changes, weight_nudges, bias_nudges = self.back_propogration(Y, m, activations, weights, weighted_sums, weighted_sum_changes, weight_nudges, bias_nudges)
            weights, biases = self.update_parameters(weights, weight_nudges, biases, bias_nudges)
            

            if i % 20 == 0:
                print(f"Iteration: {i}")
                print(f"Accuracy: {self.get_accuracy(self.get_predictions(activations[len(activations) - 1]), Y)}")
                print("")
                # print(f"Weighted Sum 1: {weighted_sums[0]}")
                # print(f"Weighted Sum 2: {weighted_sums[1]}")
                # print(f"Weighted Sum 3: {weighted_sums[2]}")
                


    def init_neural_net (self, m):    
        # Create matrices for each layer of the Neural Network
        A_0 = np.zeros((MNIST_Regression.layer_0_size, m))
        A_1 = np.zeros((MNIST_Regression.layer_1_size, m))
        A_2 = np.zeros((MNIST_Regression.layer_2_size, m))
        A_3 = np.zeros((MNIST_Regression.layer_3_size, m))

        # print("A_0 shape: ", A_0.shape)
        # print("A_1 shape: ", A_1.shape)
        # print("A_2 shape: ", A_2.shape)
        # print("A_3 shape: ", A_3.shape)

        # List of all the activation layers
        activations = [A_0, A_1, A_2, A_3]

        # Create matrices for set of weights between layers in the Neural Network
        W_1 = np.random.rand(MNIST_Regression.layer_1_size, MNIST_Regression.layer_0_size) - 0.5
        W_2 = np.random.rand(MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size) - 0.5
        W_3 = np.random.rand(MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size) - 0.5


        # print("W_1 shape: ", W_1.shape)
        # print("W_2 shape: ", W_2.shape)
        # print("W_3 shape: ", W_3.shape)

        # print(f"Weights 1: {W_1}")
        # print(f"Weights 2: {W_2}")
        # print(f"Weights 3: {W_3}")

        # List of all the weights
        weights = [W_1, W_2, W_3]
    
        # Create matrices for set of biases between layers in the Neural Network
        B_1 = np.zeros((MNIST_Regression.layer_1_size, m))
        B_2 = np.zeros((MNIST_Regression.layer_2_size, m))
        B_3 = np.zeros((MNIST_Regression.layer_3_size, m))

        # print("B_1 shape: ", B_1.shape)
        # print("B_2 shape: ", B_2.shape)
        # print("B_3 shape: ", B_3.shape)
        # List of all the biases
        biases = [B_1, B_2, B_3]

        # Create matrices for weighter sum between layers in the Neural Network
        Z_1 = np.empty((MNIST_Regression.layer_1_size, m))
        Z_2 = np.empty((MNIST_Regression.layer_2_size, m))
        Z_3 = np.empty((MNIST_Regression.layer_3_size, m))

        # print("Z_1 shape: ", Z_1.shape)
        # print("Z_2 shape: ", Z_2.shape)
        # print("Z_3 shape: ", Z_3.shape)
        # List of all the weighted sums
        weighted_sums = [Z_1, Z_2, Z_3]

        return activations, weights, biases, weighted_sums



    def init_nudges(self, m):
        # Create matrices that contain nudges to add to weights between layers
        dW_1 = np.empty((MNIST_Regression.layer_1_size, MNIST_Regression.layer_0_size))
        dW_2 = np.empty((MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size))
        dW_3 = np.empty((MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size))

        # print("dW_1 shape: ", dW_1.shape)
        # print("dW_2 shape: ", dW_2.shape)
        # print("dW_3 shape: ", dW_3.shape)
        # List of all the weight nudge matrices
        weight_nudges = [dW_1, dW_2, dW_3]

        # Create matrices that contain nudges to add to biases between layers
        dB_1 = np.empty((MNIST_Regression.layer_1_size, 1))
        dB_2 = np.empty((MNIST_Regression.layer_2_size, 1))
        dB_3 = np.empty((MNIST_Regression.layer_3_size, 1))

        # print("dB_1 shape: ", dB_1.shape)
        # print("dB_2 shape: ", dB_2.shape)
        # print("dB_3 shape: ", dB_3.shape)
        # List of all the bias nudge matrices
        bias_nudges = [dB_1, dB_2, dB_3]

        # Create matrices that contain the change in the weighted sums for dW and dB computation
        dZ_1 = np.empty((MNIST_Regression.layer_1_size, m))
        dZ_2 = np.empty((MNIST_Regression.layer_2_size, m))
        dZ_3 = np.empty((MNIST_Regression.layer_3_size, m))

        # print("dZ_1 shape: ", dZ_1.shape)
        # print("dZ_2 shape: ", dZ_2.shape)
        # print("dZ_3 shape: ", dZ_3.shape)
        # Create list of all the changes in the weighted sums
        weighted_sum_changes = [dZ_1, dZ_2, dZ_3]

        return weight_nudges, bias_nudges, weighted_sum_changes



    def load_data(self):
        datafile = pd.read_csv(MNIST_Regression.datafile_name)   # load in training data (each training sample is a row that is 785 columns long (col 1 = expected digit, col2 -> 785 is each pixel))
        data = np.array(datafile)
        np.random.shuffle(data)
        m, n = data.shape

        # set up validation data to be samples 0 - 1000
        validation_data = data[0 : 1000].T
        Y_validation = validation_data[0]
        X_validation = validation_data[1 : n]
        X_validation = X_validation / 255.

        # set up training data to be sameples 0 - last training example
        training_data = data[1000 : m].T
        Y_training = training_data[0]
        X_training = training_data[1 : n]
        X_training = X_training / 255.
        # print("Y_training shape: ", Y_training.shape)
        # print("X_training shape: ", X_training.shape)

        # print('X_validation:', X_validation)
        # print('X_training: ', X_training)
        
        return X_validation, Y_validation, X_training, Y_training



    # Forward progpagation calculates all the weighted sums for each activation layer
    def forward_propogration(self, X, activation_layers, weights, weighted_sums, biases):
        # print('FORWARD PROPOGATION:')
        # print()
        # print()
        activation_layers[0] = X
        # print(f'Activation 0: {activation_layers[0][254]}')
        
        for i in range(len(weighted_sums)):
            weighted_sums[i] = weights[i].dot(activation_layers[i]) + biases[i]
            # print(f'Activation {i + 1}: {activation_layers[i + 1][0]}')
            
            # print(f'i = {i}: {weighted_sums[i].shape} = {weights[i].dot(activation_layers[i]).shape} + {biases[i].shape}')
            if i + 1 == len(activation_layers):
                activation_layers[i + 1] = self.softmax(weighted_sums[i])
                # print(f'i = {i}: {activation_layers[i + 1].shape} = {self.softmax(weighted_sums[i]).shape}')
            else:
                activation_layers[i + 1] = self.ReLU(weighted_sums[i])
                # print(f'i = {i}: {activation_layers[i + 1].shape} = {self.ReLU(weighted_sums[i]).shape}')    
        
        # print("FORWARD PROPOGATION!!!")
        # print()
        # print(f"A_0 shape: {activation_layers[0].shape}")
        # print(f"A_1 shape: {activation_layers[1].shape}")
        # print(f"A_2 shape: {activation_layers[2].shape}")
        # print(f"A_3 shape: {activation_layers[3].shape}")

        # print(f"W_1 shape: {weights[0].shape}")
        # print(f"W_2 shape: {weights[1].shape}")
        # print(f"W_3 shape: {weights[2].shape}")

        # print(f"B_1 shape: {biases[0].shape}")
        # print(f"B_2 shape: {biases[1].shape}")
        # print(f"B_3 shape: {biases[2].shape}")
        
        # print(f"Z_1 shape: {weighted_sums[0].shape}")
        # print(f"Z_2 shape: {weighted_sums[1].shape}")
        # print(f"Z_3 shape: {weighted_sums[2].shape}")

        # print()
        # print()

        return activation_layers, weights, biases, weighted_sums



    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A


    # Sigmoid function squishes all the weighted sums between 0 - 1 for compatibility
    # with neuron activation layers
    def sigmoid(self, Z):
        Z = np.clip(Z, -600, 600)     # to prevent overflow
        return 1/(1 + np.exp(-Z))


    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def ReLU_deriv(self, Z):
        return Z > 0


    def back_propogration(self, Y, samples, activation_layers, weights, weighted_sums, weighted_sum_changes, weight_nudges, bias_nudges):
        # print("BACK PROPOGATION:")
        # print()
        # print()
        one_hot_Y = self.one_hot(Y)
        i = len(weighted_sum_changes) - 1
        while i >= 0:
            if i == len(weighted_sum_changes) - 1:
                weighted_sum_changes[i] = activation_layers[len(activation_layers) - 1] - one_hot_Y
                # print(f"i = {i}: {weighted_sum_changes[i].shape} = {activation_layers[len(activation_layers) - 1].shape} - {one_hot_Y.shape}")
                
            else:
                weighted_sum_changes[i] = np.dot(weights[i + 1].T, weighted_sum_changes[i + 1]) * self.ReLU_deriv(weighted_sums[i])
                # print(f"i = {i}: {weighted_sum_changes[i].shape} = {np.dot(weights[i + 1].T, weighted_sum_changes[i + 1]).shape} * {self.ReLU_deriv(weighted_sums[i]).shape}")

            weight_nudges[i] = (1 / samples) * np.dot(weighted_sum_changes[i], activation_layers[i].T)
            # print(f"i = {i}: {weight_nudges[i].shape} = {1 / samples} * {np.dot(weighted_sum_changes[i], activation_layers[i].T).shape}")
            bias_nudges[i] = (1 / samples) * np.sum(weighted_sum_changes[i], axis = 1).reshape(-1, 1)
            # print(f"i = {i}: {bias_nudges[i].shape} = {1 / samples} * {np.sum(weighted_sum_changes[i], axis = 1).reshape(-1, 1).shape}")

            i -= 1

        # print("BACK PROPOGATION!!!")
        # print()
        # print(f"A_0 shape: {activation_layers[0].shape}")
        # print(f"A_1 shape: {activation_layers[1].shape}")
        # print(f"A_2 shape: {activation_layers[2].shape}")
        # print(f"A_3 shape: {activation_layers[3].shape}")

        # print(f"W_1 shape: {weights[0].shape}")
        # print(f"W_2 shape: {weights[1].shape}")
        # print(f"W_3 shape: {weights[2].shape}")

        # print(f"dw_1 shape: {weight_nudges[0].shape}")
        # print(f"dw_2 shape: {weight_nudges[1].shape}")
        # print(f"dw_3 shape: {weight_nudges[2].shape}")

        # print(f"dB_1 shape: {bias_nudges[0].shape}")
        # print(f"dB_2 shape: {bias_nudges[1].shape}")
        # print(f"dB_3 shape: {bias_nudges[2].shape}")

        # print(f"dZ_1 shape: {weighted_sum_changes[0].shape}")
        # print(f"dZ_2 shape: {weighted_sum_changes[1].shape}")
        # print(f"dZ_3 shape: {weighted_sum_changes[2].shape}")
        # print()
        # print()

        return weighted_sum_changes, weight_nudges, bias_nudges

    

    def deriv_sigmoid(self, Z):
        sigmoid = self.sigmoid(Z)
        return (sigmoid * (1 - sigmoid))

    

    # take the matrix of correct answers and change each column element to be zero except for the correct answer which will be 1
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))              # fill one_hot_Y with zeros and change its size to Y.size = m rows and Y.max() + 1= 9 + 1 = 10 columns 
        one_hot_Y[np.arange(Y.size), Y] = 1                      # index through one_hot_Y by accessing row np.arange(Y.size) which is an array from 0 - m, and column Y which is a value from 0 - 9, and set it to 1
        one_hot_Y = one_hot_Y.T                                  # transpose one_hot_Y to make it compatible with matrix subtraction
        return one_hot_Y 



    def update_parameters(self, weights, weight_nudges, biases, bias_nudges):
        for i in range(len(weights)):
            weights[i] = weights[i] - MNIST_Regression.learning_ratio * weight_nudges[i]
            biases[i] = biases[i] - MNIST_Regression.learning_ratio * bias_nudges[i]
        return weights, biases

    

    def get_accuracy(self, predictions, Y):
        print(f'PREDICTED NUMBERS: {predictions[0 : 30]}')
        print(f'EXPECTED NUMBERS:  {Y[0 : 30]}')
        return np.sum(predictions == Y) / Y.size



    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)