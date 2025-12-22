import tensorflow as tf
import numpy as np
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import os

from tensorflow import keras
from keras import layers

os.environ['CUDA_VISIBLE_DEVICES'] = "0"


# IF THE MODEL IS HAVING TROUBLE PROCESSING THE DATA
# TRY DIVIDING ALL THE GREYSCALE PIXEL VALUES BY 255!!


PATH = '/home/massimo/Github/AI-Development/TensorFlow/Facial-Recognition/'

# load all images with labels
data_numpy = np.load(PATH + '/training_data/training_data.npy')

# extract labels from numpy dataset
labels = data_numpy[:, 0]

# extract greyscale image values from numpy dataset
training = data_numpy[:, 1:]

# put the labels and greyscale values into tensorflow dataset
dataset = tf.data.Dataset.from_tensor_slices((training, labels))

input_neurons = training.shape[1]

# print(input_neurons)
# print(input_neurons//10)
# print(input_neurons//50)
# print(input_neurons//500)


model = keras.Sequential([
    layers.Dense(units=input_neurons, activation='relu'),
    layers.Dense(units=input_neurons/10, activation='relu'),
    layers.Dense(units=input_neurons/50, activation='relu'),
    layers.Dense(units=input_neurons/500, activation='relu'),
    layers.Dense(units=2, activation='softmax')
])

model.compile(
    optimizer='adam', 
    loss='binary_crossentropy', 
    metrics=['accuracy'],
    run_eagerly=False
)


