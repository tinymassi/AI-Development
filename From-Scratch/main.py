# ********** HOW TO RUN NEURAL NETWORK: ********** #
# In terminal window, type: python3 main.cpp       #
# and press enter.                                 #
# ************************************************ #

from MNIST_Regression import MNIST_Regression

hidden_layers = [18, 18]
AI_model = MNIST_Regression(0.5, 10000, 10000, hidden_layers, 100, 1)