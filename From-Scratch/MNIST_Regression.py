import numpy as np

class MNIST_Regression:
    def __init__(self):

    
    dim_pixels = 28*28                         # w * h of the input pixel grid
    m = 10000                                  # m = number of training examples

    layer_0_size = dim_pixels                  # input layer has 784 neurons (one for each pixel)
    layer_1_size = 16                          # hidden layer 1 has 16 neurons
    layer_2_size = 16                          # hidden layer 2 has 16 neurons
    layer_3_size = 10                          # output layer has 10 neurons (one for each predicted digit 1 - 9)

    T = np.empty(dim_pixels, m)

    # Create matrices for each layer of the Neural Network
    A_0 = np.empty(layer_0_size, m)
    A_1 = np.empty(layer_1_size, m)
    A_2 = np.empty(layer_2_size, m)
    A_3 = np.empty(layer_3_size, m)

    # Create matrices for set of weights between layers in the Neural Network
    W_1 = np.empty(layer_1_size, layer_0_size)
    W_2 = np.empty(layer_2_size, layer_1_size)
    W_3 = np.empty(layer_3_size, layer_2_size)

    
