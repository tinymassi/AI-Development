# ********** HOW TO RUN NEURAL NETWORK: ********** #
# In terminal window, type: python3 main.cpp       #
# and press enter.                                 #
# ************************************************ #

from MNIST_Regression import MNIST_Regression
import numpy as np
import pandas as pd


datafile_name = 'train.csv.zip'
datafile = pd.read_csv(datafile_name)
data = (np.array(datafile)).T


hidden_layers = [18, 18]
AI_model = MNIST_Regression(1, 10000, 10000, hidden_layers, 200, 1, False)


user_input = ''
while user_input.lower != 'end':
    user_input = input(">> ")

    if user_input.lower == 'end':
        print('Thanks for using the program!')
    else:
        col = int(user_input)

        if col < 0 or col > data.shape[1]:
            print(f'Invalid index. Pick a number between 0 and {data.shape[1]}')
        else:
            number = data[1:, col:col+1]
            number = number / 255.
            print(f'NUMBER: {data[0:1, col][0]}')
            print(f'PREDICTION: {AI_model.read_number(number)}')