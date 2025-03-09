# For importing data
import pandas as pd                                                                 # python data manipulation library
from sklearn.model_selection import train_test_split

# For importing dependencies
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model                          # our core model is a sequntial model. Load model allows us to load model from memory later
from tensorflow.keras.layers import Dense                                           # we will be using a dense, fully connected layer in our NN
from sklearn.metrics import accuracy_score                                          # accuracy score = metric to see how well model performs

import matplotlib.pyplot as plt


class runAI:

    def __init__(self):
        self.import_data()
        self.build_model()
        self.compile_model()
        self.predict()
        self.save_model()

    def import_data(self):
        datafile = pd.read_csv('Churn.csv')                                                                               # read data from csv data file

        self.X = pd.get_dummies(datafile.drop(['Churn', 'Customer ID'], axis=1))                                          # select data from columns 'Customer ID' -> 'Churn'? set as X
        self.y = datafile['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)                                                # select data from column 'Churn' and set as y. If data = yes, then y = 1. Otherwise, y = 0

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size = 0.1)          # creates data trains for training data from csv and predictive results
        self.y_train.head()
        return

    def build_model(self):
        self.model = Sequential()                                                                                         # instantiating the sequential model class.
        self.model.add(Dense(units = 16, kernel_regularizer = keras.regularizers.l2(0.001), activation = 'relu', input_dim = len(self.X_train.columns)))                     # dense fully connected hidden layer with 32 nodes/neurons. Activation = modifier for NN output 'relu' converts raw data to value from 0 -> infinity. input_dim = same number of dimensions as dataframe
        self.model.add(Dense(units = 32, kernel_regularizer = keras.regularizers.l2(0.001), activation = 'relu'))                                                            # dense fully connected hidden layer with 64 nodes/neurons
        self.model.add(Dense(units = 1, activation = 'sigmoid'))                                                          # last layer that only has 1 output. 'Yes' or 'No'. Trying to predict whether or not a client has churned. 'Sigmoid' means that it takes the output from previous layer and converts it to either a 1 ('Yes') or 0 ('No')
        return
    
    def compile_model(self):
        self.model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])                             # telling tensorflow how to train the model. loss means how far we are from our desired output. optimizer means how we can optimize how to get to the desired true outcome from the prediction. metrics will tell us how well the model is performing
        return
    
    def predict(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs = 1000, batch_size = 32, validation_data = (self.X_test, self.y_test))                         # this line is for training the model. X_train = features y_train = target output. epochs = how long to train for. batch size = how long of a batch to pass through the tensorflow before making an update
        self.y_hat = self.model.predict(self.X_test)                                                                      # run a prediction for the X_test dataframe
        self.y_hat = [0 if val < 0.5 else 1 for val in self.y_hat]                                                        # convert the output in y_hat to 1 or 0 based on if the data is less than 0.5 or greater than 0.5
        accuracy_score(self.y_test, self.y_hat)                                                                           # calculates the final accuracy score. training longer, use regularization, pre-processing on data may be needed for better results/accuracy
        
        fig, (ax1, ax2) = plt.subplots(2, 1)  # 2 rows, 1 column 

        ax1.plot(self.history.history['accuracy'], color='b', label = 'Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], color='r', label = 'Validation Accuracy')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.legend()
        

        ax2.plot(self.history.history['loss'], color='b', label = 'Training loss')
        ax2.plot(self.history.history['val_loss'], color='r', label = 'Validation loss')
        ax2.set_xlabel('Epochs')
        ax2.set_ylabel('Loss')
        ax2.set_title('Training and Validation loss');
        ax2.legend()

        plt.tight_layout()

        return
    
    def save_model(self):
        plt.show()
        self.model.save('tfmodel.keras')                                                                                  # saves the model
        del self.model                                                                                                    # remove the model from memory
        self.model = load_model('tfmodel.keras')                                                                          # reload the model from memory for use
        return



    
