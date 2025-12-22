# *************************** PROGRAM EXPLANATION AND AUTHOR *************************** #
# This is a convolutional neural network designed to recognize hand drawn digits on a    #
# 28x28 pixel grid. It uses the MNIST Database which is comprised of tens of thousands   #
# of labeled hand drawn digits. This network has 4 activation layers: 1 input layer, 2   #
# hidden layers, and 1 output layer. This Network does not use pytorch, keras, or        #
# tensorflow. Just numpy, multivariable calculus, and linear algebra.                    #
#                                                                                        #
# Author: Massimo Ginella                                                                #
# Email: massimoginella12@gmail.com                                                      #
# ************************************************************************************** #


# ********************************** MORE INFORMATION ********************************** #
# This code is largely inspired by '3Blue1Brown' and 'Samson Zhang' on YouTube           #
#                                                                                        #
# 3Blue1Brown's Video: https://www.youtube.com/watch?v=aircAruvnKk                       #                                           
# Samson Zhang's Video: https://www.youtube.com/watch?v=w8yWXqWQYmU&t=668s               #
# ************************************************************************************** #


# Import Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator



class MNIST_Regression:

    # ********************************************* #
    # CUSTOMIZE NEURAL NETWORK SPECIFICATIONS HERE: #
    # ********************************************* #

    # Neural network neuron count in first and last layer layer_0 should always = 0, layer_3 should always = 10. 
    layer_0_size = 0
    layer_3_size = 10

    # Name of input file containing training and validation data
    datafile_name = 'train.csv.zip'


    # Initializes the MNIST_Regression class
    def __init__(self, learning_ratio=0.1, validation_end=10000, training_start=10000, hidden_layers=[18, 18], num_epochs=500, epoch_step=10):

        self.datafile = pd.read_csv(self.datafile_name)
        self.data = np.array(self.datafile)
        m = self.data.shape[0]            

        # ADD ASSERTS HERE
        assert learning_ratio > 0, "Learning ratio must be non-negative. Pick a value between 0.1 - 3."
        assert validation_end <= training_start, "Validation end value must be less than or equal to the start value for the training data."
        assert validation_end > 0 and training_start > 0 and validation_end < m and training_start < m, f"Validation end value and training start value must be > 0 and less than the number of labeled images ({m})."
        assert len(hidden_layers) == 2, "Number of hidden layers must be equal to 2 since this network is only designed with 2 hidden layers in mind."
        assert hidden_layers[0] > 1 and hidden_layers[1] > 1, "Number of neurons in each hidden layer cannot be less than or equal to 1."
        assert num_epochs > 0, "Number of training epochs must be greater than 0."
        assert epoch_step > 0, "Epoch step must be greater than 0."

        # Neural network neuron count in hidden layers. Only for changing layer_1 and layer_2
        self.layer_1_size = hidden_layers[0]
        self.layer_2_size = hidden_layers[1]
        
        # Ratio for the rate at which the model learns
        self.num_epochs = num_epochs

        # Data samples used for validation (0 -> validation_end)
        self.validation_end = validation_end
        # Starting point for data samples used for training (training_start -> m)
        self.training_start = training_start

        # Ratio for the rate at which the model learns
        self.learning_ratio = learning_ratio

        # Number of steps that are displayed between epochs
        self.epoch_step = epoch_step

        # Load data into respective training and labeled data
        X_validation, Y_validation, X_training, Y_training = self.load_data()

        # Set the number of neurons in the first layer to the w x h pixel dimension of input img
        self.layer_0_size = X_validation.shape[0]

        # Run the neural network with the aquired training and label data
        self.gradient_descent(X_validation, Y_validation, X_training, Y_training, self.num_epochs)



    # Gradient descent function runs forward propogation, back propogation and updates the parameters (weights and biases) of the network
    def gradient_descent(self, X_val, Y_val, X_train, Y_train, epochs):

        # Initialize the amount of training samples (m) and sizes of the activation layers, weights, biases, and weighted sums based upon it
        m = X_train.shape[1] 
        activations, weights, biases, weighted_sums = self.init_neural_net(m)
        weight_nudges, bias_nudges, weighted_sum_changes = self.init_nudges(m)

        indexes = []
        training_accuracy_scores = []
        validation_accuracy_scores = []
    

        for i in range(epochs + 1):
            activations, weighted_sums = self.forward_propogration(X_train, activations, weights, weighted_sums, biases)
            weighted_sum_changes, weight_nudges, bias_nudges = self.back_propogration(Y_train, m, activations, weights, weighted_sums, weighted_sum_changes, weight_nudges, bias_nudges)
            weights, biases = self.update_parameters(weights, weight_nudges, biases, bias_nudges)
            

            training_predictions, training_actual, training_accuracy = self.get_accuracy(self.get_predictions(activations[len(activations) - 1]), Y_train)
            training_accuracy = training_accuracy * 100
            if i % self.epoch_step == 0:
                print(f"EPOCH: {i} / {self.num_epochs}")
                print(f'NEURAL NET PREDICTED NUMBERS: {training_predictions}')
                print(f'ACTUAL NUMBERS:               {training_actual}')
                print(f"Training Accuracy: {training_accuracy:.2f}%")
                print("")


            validation_activations, validation_weighted_sums = self.forward_propogration(X_val, activations, weights, weighted_sums, biases)
            validation_predictions, validation_actual, validation_accuracy = self.get_accuracy(self.get_predictions(activations[len(activations) - 1]), Y_val)
            validation_accuracy = validation_accuracy * 100 


            indexes.append(i)
            training_accuracy_scores.append(training_accuracy)
            validation_accuracy_scores.append(validation_accuracy)

        print("")    
        print('Training and validation done!')
        self.plot(validation_accuracy_scores, training_accuracy_scores, indexes, X_val, Y_val, validation_activations, m)



    # Initializes the sizes of the activation, weight, bias, and weighted sum matrices
    def init_neural_net (self, m):
        # Create matrices for each layer of the Neural Network
        A_0 = np.zeros((self.layer_0_size, m))
        A_1 = np.zeros((self.layer_1_size, m))
        A_2 = np.zeros((self.layer_2_size, m))
        A_3 = np.zeros((self.layer_3_size, m))

        # List of all the activation layers
        activations = [A_0, A_1, A_2, A_3]

        # Create matrices for set of weights between layers in the Neural Network
        W_1 = np.random.rand(self.layer_1_size, self.layer_0_size) - 0.5
        W_2 = np.random.rand(self.layer_2_size, self.layer_1_size) - 0.5
        W_3 = np.random.rand(self.layer_3_size, self.layer_2_size) - 0.5
    
        weights = [W_1, W_2, W_3]
    
        # Create matrices for set of biases between layers in the Neural Network
        B_1 = np.zeros((self.layer_1_size, 1))
        B_2 = np.zeros((self.layer_2_size, 1))
        B_3 = np.zeros((self.layer_3_size, 1))

        # List of all the biases
        biases = [B_1, B_2, B_3]

        # Create matrices for weighter sum between layers in the Neural Network
        Z_1 = np.empty((self.layer_1_size, m))
        Z_2 = np.empty((self.layer_2_size, m))
        Z_3 = np.empty((self.layer_3_size, m))

        # List of all the weighted sums
        weighted_sums = [Z_1, Z_2, Z_3]

        return activations, weights, biases, weighted_sums



    # Initializes the size of the weight, bias, and weighted sum nudge matrices for param updating
    def init_nudges(self, m):
        # Create matrices that contain nudges to add to weights between layers
        dW_1 = np.empty((self.layer_1_size, self.layer_0_size))
        dW_2 = np.empty((self.layer_2_size, self.layer_1_size))
        dW_3 = np.empty((self.layer_3_size, self.layer_2_size))

        # List of all the weight nudge matrices
        weight_nudges = [dW_1, dW_2, dW_3]

        # Create matrices that contain nudges to add to biases between layers
        dB_1 = np.empty((self.layer_1_size, 1))
        dB_2 = np.empty((self.layer_2_size, 1))
        dB_3 = np.empty((self.layer_3_size, 1))

        # List of all the bias nudge matrices
        bias_nudges = [dB_1, dB_2, dB_3]

        # Create matrices that contain the change in the weighted sums for dW and dB computation
        dZ_1 = np.empty((self.layer_1_size, m))
        dZ_2 = np.empty((self.layer_2_size, m))
        dZ_3 = np.empty((self.layer_3_size, m))

        # Create list of all the changes in the weighted sums
        weighted_sum_changes = [dZ_1, dZ_2, dZ_3]

        return weight_nudges, bias_nudges, weighted_sum_changes



    # Loads data 
    def load_data(self):
        # load in training data (each training sample is a row that is 785 column long where column 1 = labeled digit, column 2 -> column 785 = each pixel)
        print("Fetching Data...")
        np.random.shuffle(self.data)
        m, n = self.data.shape


        # set up validation data to be samples 0 - validation end
        validation_data = self.data[0 : self.validation_end].T
        Y_validation = validation_data[0]
        X_validation = validation_data[1 : n]
        X_validation = X_validation / 255.
        print("Validation Data Loaded.")

        # set up training data to be sameples training start - last training example
        training_data = self.data[self.training_start : m].T
        Y_training = training_data[0]
        X_training = training_data[1 : n]
        X_training = X_training / 255.
        print("Training Data Loaded.")
        print()
        
        return X_validation, Y_validation, X_training, Y_training



    # Forward progpagation calculates all the weighted sums for each activation layer
    def forward_propogration(self, X, activation_layers, weights, weighted_sums, biases):
        activation_layers[0] = X
        
        for i in range(len(weighted_sums)):
            weighted_sums[i] = weights[i].dot(activation_layers[i]) + biases[i]

            if i + 1 == len(activation_layers):
                activation_layers[i + 1] = self.softmax(weighted_sums[i])
            else:
                activation_layers[i + 1] = self.sigmoid(weighted_sums[i])

        return activation_layers, weighted_sums



    # Converts raw data into probabilities 
    def softmax(Z):
        A = np.exp(Z) / sum(np.exp(Z))
        return A



    # Squishes all the weighted sums between 0 & 1
    def sigmoid(self, Z):
        Z = np.clip(Z, -600, 600)     # to prevent overflow
        return 1/(1 + np.exp(-Z))


    
    # Derivative of sigmoid function
    def deriv_sigmoid(self, Z):
        sigmoid = self.sigmoid(Z)
        return (sigmoid * (1 - sigmoid))



    # Sets weighted sums to either 0 or 1
    def ReLU(self, Z):
        return np.maximum(Z, 0)



    # Derivative of ReLU function
    def ReLU_deriv(self, Z):
        return Z > 0



    # Calculates the nudges needed to be added to the weights and biases of the neural network in order to optimize it for a certian input pixel pattern
    def back_propogration(self, Y, samples, activation_layers, weights, weighted_sums, weighted_sum_changes, weight_nudges, bias_nudges):
        one_hot_Y = self.one_hot(Y)
        i = len(weighted_sum_changes) - 1
        while i >= 0:
            if i == len(weighted_sum_changes) - 1:
                weighted_sum_changes[i] = activation_layers[len(activation_layers) - 1] - one_hot_Y
            else:
                weighted_sum_changes[i] = np.dot(weights[i + 1].T, weighted_sum_changes[i + 1]) * self.deriv_sigmoid(weighted_sums[i])

            weight_nudges[i] = (1 / samples) * np.dot(weighted_sum_changes[i], activation_layers[i].T)
            bias_nudges[i] = (1 / samples) * np.sum(weighted_sum_changes[i], axis = 1).reshape(-1, 1)

            i -= 1

        return weighted_sum_changes, weight_nudges, bias_nudges

    

    # Takes the matrix of labeled data and changes each column element to be zero except for the correct answer which will be 1
    def one_hot(self, Y):
        one_hot_Y = np.zeros((Y.size, Y.max() + 1))              # fill one_hot_Y with zeros and change its size to Y.size = m rows and Y.max() + 1= 9 + 1 = 10 columns 
        one_hot_Y[np.arange(Y.size), Y] = 1                      # index through one_hot_Y by accessing row np.arange(Y.size) which is an array from 0 - m, and column Y which is a value from 0 - 9, and set it to 1
        one_hot_Y = one_hot_Y.T                                  # transpose one_hot_Y to make it compatible with matrix subtraction
        return one_hot_Y 



    # Updates the weights and biases of the neural network based upon the weight and bias nudges calculated in back propogation
    def update_parameters(self, weights, weight_nudges, biases, bias_nudges):
        for i in range(len(weights)):
            weights[i] = weights[i] - self.learning_ratio * weight_nudges[i]
            biases[i] = biases[i] - self.learning_ratio * bias_nudges[i]
        return weights, biases

    

    # Calculates the accuracy of the neural networks predictions in the final output layer (also returns a slice of the predictiona and labels for visualization)
    def get_accuracy(self, predictions, Y):
        return predictions[0 : 30], Y[0 : 30], np.sum(predictions == Y) / Y.size



    # Takes raw output layer neurons and creates a numpy array of each numerical guess from the ouput layer data
    def get_predictions(self, output_layer):
        return np.argmax(output_layer, 0)



    # Plots relevant data at the end of training such as accuracy improvements over time, final accuracy scores, and visualizations of the neural networks correct and incorrect predictions
    def plot(self, validation_accuracy_scores, training_accuracy_scores, epochs, X_val, Y_val, validation_activations, m):
        # Create a plt plot that is a 2 x 2 graph layout of size 12 x 8
        fig1, ax = plt.subplots(2, 2, figsize = (12, 8))


        # Plot the training and validation accuracy comparison graphs
        ax[0 , 0].plot(epochs, training_accuracy_scores, label = 'Training Accuracy', c = 'b')
        ax[0 , 0].plot(epochs, validation_accuracy_scores, label = 'Validation Accuracy', c = 'r')
        ax[0 , 0].set_ylim(0, 100)
        ax[0 , 0].set_xlim(0, self.num_epochs)
        ax[0 , 0].xaxis.set_major_locator(MultipleLocator(100))
        ax[0 , 0].yaxis.set_major_locator(MultipleLocator(10))
        ax[0 , 0].set_title('Neural Network Accuracy:')
        ax[0 , 0].set_xlabel('Epochs')
        ax[0 , 0].set_ylabel('Accuracy')
        ax[0 , 0].text(0.001, 12, f'Neurons/Layer: L1: {self.layer_0_size}, L2: {self.layer_1_size}, L3: {self.layer_2_size}, L4: {self.layer_3_size} ', fontsize = 8)
        ax[0 , 0].text(0.001, 7, f'Learning Ratio: {self.learning_ratio}', fontsize = 8)
        ax[0 , 0].text(0.001, 2, (f'Val Data Size: {self.validation_end}, Train Data Size: {m - self.training_start}'), fontsize = 8)
        ax[0 , 0].legend()


        # Plot the final accuracy values for the training and validation processeses
        categories = ['Validation', 'Training']
        data = [validation_accuracy_scores[-1], training_accuracy_scores[-1]]
        ax[1 , 0].bar(categories, data)
        ax[1 , 0].set_ylim(0, 100)
        ax[1 , 0].yaxis.set_major_locator(MultipleLocator(10))
        ax[1 , 0].set_title('Validation vs Training Accuracy')
        ax[1 , 0].set_ylabel('Final Accuracy Scores')
        ax[1 , 0].text(.9, 93, f'{training_accuracy_scores[-1]:.2f}%', fontsize = 12)
        ax[1 , 0].text(-.08, 93, f'{validation_accuracy_scores[-1]:.2f}%', fontsize = 12)


        # Plot a visualization of a correct prediction
        predictions = self.get_predictions(validation_activations[-1])
        correct_prediction = 0

        for i in range(len(predictions)):
            if predictions[i] == Y_val[i]:
                correct_prediction = i
                break

        current_image = X_val[:, correct_prediction, None]
        prediction = predictions[correct_prediction]
        label = Y_val[correct_prediction]
        current_image = current_image.reshape((28, 28)) * 255

        ax[0 , 1].set_title('Correct Prediction Example:', color='green')
        ax[0 , 1].imshow(current_image, cmap = 'gray', interpolation='nearest')
        ax[0 , 1].text(0, 1, f'Actual Number: {label}', color='white', fontsize = 12)
        ax[0 , 1].text(0, 2.5, f'Predicted Number: {prediction}', color='green', fontsize = 12)

        
        # Plot a visualization of an incorrect prediction
        incorrect_prediction = 0
        for i in range(len(predictions)):
            if predictions[i] != Y_val[i]:
                incorrect_prediction = i
                break

        current_image = X_val[:, incorrect_prediction, None]
        prediction = predictions[incorrect_prediction]
        label = Y_val[incorrect_prediction]
        current_image = current_image.reshape((28, 28)) * 255
        
        ax[1 , 1].set_title('Incorrect Prediction Example:', color='red')
        ax[1 , 1].imshow(current_image, cmap = 'gray', interpolation='nearest')
        ax[1 , 1].text(0, 1, f'Actual Number: {label}', color='white', fontsize=12)
        ax[1 , 1].text(0, 2.5, f'Predicted Number: {prediction}', color='red', fontsize=12)

        fig1.tight_layout()

        plt.show()

