import numpy as np

class MNIST_Regression:

    def __init__(self):
        print('place holder')


    # Forward progpagation calculates all the weighted sums for each activation layer
    def forward_propogration(self):
        for i in range(len(self.activation_layers)):
            self.weighted_sums[i] = np.dot(self.weights[i], self.activation_layers[i]) + self.biases[i]
            self.activation_layers[i + 1] = self.sigmoid(self.weighted_sums[i])        
        return


    # Sigmoid function squishes all the weighted sums between 0 - 1 for compatibility
    # with neuron activation layers
    def sigmoid(self, Z):
        return 1/(1 + np.exp(-self.Z))


    def back_propogration(self):
        i = len(self.weighted_sum_changes) - 1
        while i >= 0:
            if i == len(self.weighted_sum_changes) - 1:
                self.weighted_sum_changes[i] = self.activation_layers[len(self.activation_layers) - 1] - self.Y
            else:
                self.weighted_sum_changes[i] = np.dot(self.weights[i + 1].T, self.weighted_sum_changes[i + 1], )# WHAT IS G PRIME OF Z??

            self.weight_nudges[i] = (1 / self.m) * np.dot(self.weighted_sum_changes[i], self.activation_layers[i].T)
            self.bias_nudges[i]= (1 / self.m) * np.sum(self.weighted_sum_changes, axis = 1)
        return


    def update_parameters(self):
        for i in range(len(self.weights)):
            self.weights[i] = self.weights[i] - (self.learning_ratio * self.weight_nudges[i])
            self.biases[i] = self.biases[i] - (self.learning_ratio * self.bias_nudges[i])
        return


    dim_pixels = 28*28                         # w * h of the input pixel grid
    m = 10000                                  # m = number of training samples
    learning_ratio = 0

    layer_0_size = dim_pixels                  # input layer has 784 neurons (one for each pixel)
    layer_1_size = 16                          # hidden layer 1 has 16 neurons
    layer_2_size = 16                          # hidden layer 2 has 16 neurons
    layer_3_size = 10                          # output layer has 10 neurons (one for each predicted digit 1 - 9)

    # Matrix that contains dim_pixel row and m columns
    T = np.empty(dim_pixels, m)

    # Matrix that contains correct answer choice from training data for comparison to predicted output
    Y = np.empty(layer_3_size, m)

    # Create matrices for each layer of the Neural Network
    A_0 = np.empty(layer_0_size, m)
    A_1 = np.empty(layer_1_size, m)
    A_2 = np.empty(layer_2_size, m)
    A_3 = np.empty(layer_3_size, m)

    # List of all the layers
    activation_layers = [A_0, A_1, A_2, A_3]

    # Create matrices for set of weights between layers in the Neural Network
    W_1 = np.random.uniform(-0.5, 0.5, size = (layer_1_size, layer_0_size))
    W_2 = np.random.uniform(-0.5, 0.5, size = (layer_2_size, layer_1_size))
    W_3 = np.random.uniform(-0.5, 0.5, size = (layer_3_size, layer_2_size))

    # List of all the weights matrices
    weights = [W_1, W_2, W_3]

    # Create matrices that contain nudges to add to weights between layers
    dW_1 = np.empty(layer_1_size, layer_2_size)
    dW_2 = np.empty(layer_2_size, layer_1_size)
    dW_3 = np.empty(layer_3_size, layer_2_size)

    # List of all the weight nudge matrices
    weight_nudges = [dW_1, dW_2, dW_3]

    # Create matrices for set of biases between layers in the Neural Network
    B_1 = np.zeros(layer_1_size, m)
    B_2 = np.zeros(layer_2_size, m)
    B_3 = np.zeros(layer_3_size, m)

    # List of all the biases
    biases = [B_1, B_2, B_3]

    # Create matrices that contain nudges to add to biases between layers
    dB_1 = np.empty(layer_1_size, 1)
    dB_2 = np.empty(layer_2_size, 1)
    dB_3 = np.empty(layer_3_size, 1)

    # List of all the bias nudge matrices
    bias_nudges = [dB_1, dB_2, dB_3]

    # Create matrices for weighter sum between layers in the Neural Network
    Z_1 = np.empty(layer_1_size, m)
    Z_2 = np.empty(layer_2_size, m)
    Z_3 = np.empty(layer_3_size, m)

    # List of all the weighted sums
    weighted_sums = [Z_1, Z_2, Z_3]

    # Create matrices that contain the change in the weighted sums for dW and dB computation
    dZ_1 = np.empty(layer_1_size, m)
    dZ_2 = np.empty(layer_2_size, m)
    dZ_3 = np.empty(layer_3_size, m)

    # Create list of all the changes in the weighted sums
    weighted_sum_changes = [dZ_1, dZ_2, dZ_3]



