import numpy as np

class MNIST_Regression:
    def __init__(self):

    
    def forward_propogration(self):

    
    dim_pixels = 28*28                         # w * h of the input pixel grid
    m = 10000                                  # m = number of training examples

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

    layers = [A_0, A_1, A_2, A_3]

    # Create matrices for set of weights between layers in the Neural Network
    W_1 = np.empty(layer_1_size, layer_0_size)
    W_2 = np.empty(layer_2_size, layer_1_size)
    W_3 = np.empty(layer_3_size, layer_2_size)

    weights = [W_1, W_2, W_3]

    # Create matrices for set of biases between layers in the Neural Network
    B_1 = np.empty(layer_1_size, m)
    B_2 = np.empty(layer_2_size, m)
    B_3 = np.empty(layer_3_size, m)

    biases = [B_1, B_2, B_3]

    # Create matrices for weighter sum between layers in the Neural Network
    Z_1 = np.empty(layer_1_size, m)
    Z_2 = np.empty(layer_2_size, m)
    Z_3 = np.empty(layer_3_size, m)

    weighted_sums = [Z_1, Z_2, Z_3]




