import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MNIST_Regression:

    layer_0_size = 0                           # input layer has 784 neurons (one for each pixel)
    layer_1_size = 30                          # hidden layer 1 has 16 neurons
    layer_2_size = 30                          # hidden layer 2 has 16 neurons
    layer_3_size = 10                          # output layer has 10 neurons (one for each predicted digit 1 - 9)

    learning_ratio = 0.5                       # ratio for the rate at which the model learns

    validation_end = 20000                     # how many data examples will be used for validation (0 -> validation_end)

    training_start = 20000                     # starting point for data samples used for training (training_start -> m)

    datafile_name = 'train.csv.zip'            # name of the input training and validation data file

    num_epochs = 20

    def __init__(self):

        X_validation, Y_validation, X_training, Y_training = self.load_data()

        MNIST_Regression.layer_0_size = X_validation.shape[0]

        # TRAINING
        self.gradient_descent(X_validation, Y_validation, X_training, Y_training, MNIST_Regression.num_epochs)



    def gradient_descent(self, X_val, Y_val, X_train, Y_train, epochs):
        m = X_train.shape[1] 
        activations, weights, biases, weighted_sums = self.init_neural_net(m)
        weight_nudges, bias_nudges, weighted_sum_changes = self.init_nudges(m)

        indexes = []
        training_accuracy_scores = []
        # predictions = []
        # labels = []
        # current_image = []

        for i in range(epochs + 1):
            activations, weights, biases, weighted_sums = self.forward_propogration(X_train, activations, weights, weighted_sums, biases)
            weighted_sum_changes, weight_nudges, bias_nudges = self.back_propogration(Y_train, m, activations, weights, weighted_sums, weighted_sum_changes, weight_nudges, bias_nudges)
            weights, biases = self.update_parameters(weights, weight_nudges, biases, bias_nudges)
            

            training_accuracy = (self.get_accuracy(self.get_predictions(activations[len(activations) - 1]), Y_train) * 100)
            print(f"Epoch: {i}")
            print(f"Training Accuracy: {training_accuracy:.2f}%")
            print("")

            # prediction, label, current_image = self.test_prediction(self, i, X, Y, weights, biases)

            indexes.append(i)
            training_accuracy_scores.append(training_accuracy)

        
        activations, weights, biases, weighted_sums = self.forward_propogration(X_val, activations, weights, weighted_sums, biases)
        validation_accuracy = (self.get_accuracy(self.get_predictions(activations[len(activations) - 1]), Y_val) * 100)
        print(f"Validation Accuracy: {validation_accuracy}")
        

        self.plot(validation_accuracy, training_accuracy, training_accuracy_scores, indexes)

    def init_neural_net (self, m):    
        # Create matrices for each layer of the Neural Network
        A_0 = np.zeros((MNIST_Regression.layer_0_size, m))
        A_1 = np.zeros((MNIST_Regression.layer_1_size, m))
        A_2 = np.zeros((MNIST_Regression.layer_2_size, m))
        A_3 = np.zeros((MNIST_Regression.layer_3_size, m))

        # List of all the activation layers
        activations = [A_0, A_1, A_2, A_3]

        # Create matrices for set of weights between layers in the Neural Network
        W_1 = np.random.rand(MNIST_Regression.layer_1_size, MNIST_Regression.layer_0_size) - 0.5
        W_2 = np.random.rand(MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size) - 0.5
        W_3 = np.random.rand(MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size) - 0.5
    
        weights = [W_1, W_2, W_3]
    
        # Create matrices for set of biases between layers in the Neural Network
        B_1 = np.zeros((MNIST_Regression.layer_1_size, 1))
        B_2 = np.zeros((MNIST_Regression.layer_2_size, 1))
        B_3 = np.zeros((MNIST_Regression.layer_3_size, 1))

        # List of all the biases
        biases = [B_1, B_2, B_3]

        # Create matrices for weighter sum between layers in the Neural Network
        Z_1 = np.empty((MNIST_Regression.layer_1_size, m))
        Z_2 = np.empty((MNIST_Regression.layer_2_size, m))
        Z_3 = np.empty((MNIST_Regression.layer_3_size, m))

        # List of all the weighted sums
        weighted_sums = [Z_1, Z_2, Z_3]

        return activations, weights, biases, weighted_sums



    def init_nudges(self, m):
        # Create matrices that contain nudges to add to weights between layers
        dW_1 = np.empty((MNIST_Regression.layer_1_size, MNIST_Regression.layer_0_size))
        dW_2 = np.empty((MNIST_Regression.layer_2_size, MNIST_Regression.layer_1_size))
        dW_3 = np.empty((MNIST_Regression.layer_3_size, MNIST_Regression.layer_2_size))

        # List of all the weight nudge matrices
        weight_nudges = [dW_1, dW_2, dW_3]

        # Create matrices that contain nudges to add to biases between layers
        dB_1 = np.empty((MNIST_Regression.layer_1_size, 1))
        dB_2 = np.empty((MNIST_Regression.layer_2_size, 1))
        dB_3 = np.empty((MNIST_Regression.layer_3_size, 1))

        # List of all the bias nudge matrices
        bias_nudges = [dB_1, dB_2, dB_3]

        # Create matrices that contain the change in the weighted sums for dW and dB computation
        dZ_1 = np.empty((MNIST_Regression.layer_1_size, m))
        dZ_2 = np.empty((MNIST_Regression.layer_2_size, m))
        dZ_3 = np.empty((MNIST_Regression.layer_3_size, m))

        # Create list of all the changes in the weighted sums
        weighted_sum_changes = [dZ_1, dZ_2, dZ_3]

        return weight_nudges, bias_nudges, weighted_sum_changes



    def load_data(self):
        print("Fetching Data...")
        datafile = pd.read_csv(MNIST_Regression.datafile_name)   # load in training data (each training sample is a row that is 785 columns long (col 1 = expected digit, col2 -> 785 is each pixel))
        data = np.array(datafile)
        np.random.shuffle(data)
        m, n = data.shape

        # set up validation data to be samples 0 - 1000
        validation_data = data[0 : MNIST_Regression.validation_end].T
        Y_validation = validation_data[0]
        X_validation = validation_data[1 : n]
        X_validation = X_validation / 255.
        print("Validation Data Loaded...")

        # set up training data to be sameples 0 - last training example
        training_data = data[MNIST_Regression.training_start : m].T
        Y_training = training_data[0]
        X_training = training_data[1 : n]
        X_training = X_training / 255.
        print("Training Data Loaded...")
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


    
    def plot(self, validation_accuracy, training_accuracy, accuracy_scores, epochs):
        fig, ax = plt.subplots(2, 1)

        ax[0].plot(epochs, accuracy_scores)
        ax[0].set_ylim(0, 100)
        ax[0].set_xlim(0, MNIST_Regression.num_epochs)
        ax[0].set_title('Neural Net Training Accuracy')
        ax[0].set_xlabel('Epochs')
        ax[0].set_ylabel('Training Accuracy')

        categories = ['Validation', 'Training']
        data = [validation_accuracy, training_accuracy]
        ax[1].bar(categories, data)
        ax[1].set_ylim(0, 100)
        ax[1].set_title('Validation vs Training Accuracy')
        ax[1].set_ylabel('Final Accuracy Scores')
        # define your x and y arrays to be plotted
        # plots = [(t,y1), (t,y2), (t,y3)]

        fig.tight_layout()

        # # now the real code :) 
        # curr_pos = 0

        # ax.cla()
        # ax.plot(plots[curr_pos][0], plots[curr_pos][1])
        # fig.canvas.draw()

        # fig = plt.figure()
        # fig.canvas.mpl_connect('key_press_event', self.key_event(plots))
        # ax = fig.add_subplot(111)
        # ax.plot(t,y1)
        # plt.show()

        # ani = FuncAnimation 

        plt.show()



    # def key_event(e, plots):
    #     global curr_pos

    #     if e.key == "right":
    #         curr_pos = curr_pos + 1
    #     elif e.key == "left":
    #         curr_pos = curr_pos - 1
    #     else:
    #         return
    #     curr_pos = curr_pos % len(plots)



    # def make_predictions(self, activation_layers, X, weights, biases):
    #     _, _, _, A2 = self.forward_propogration(weights, biases)
    #     predictions = self.get_predictions(A2)
    #     return predictions



    # def test_prediction(self, index, activation_layers, X_train, Y_train, weights, biases):
    #     current_image = X_train[:, index, None]
    #     prediction = self.make_predictions(activation_layers, X_train[:, index, None], weights, biases)
    #     label = Y_train[index]

    #     print("Prediction: ", prediction)
    #     print("Label: ", label)
        
    #     current_image = current_image.reshape((28, 28)) * 255
    #     plt.gray()
    #     plt.imshow(current_image, interpolation='nearest')
    #     plt.show()

    #     return prediction, label, current_image