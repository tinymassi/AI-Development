Hello!

This is my hand coded Neural Network project. It is designed to recognize hand drawn digits and classify them. I did not use numpy nor tensorflow to create this project. Only numpy and pandas. 

For this program to work, you must install the following:
- Python3
- Matplotlib
- Pandas
- Numpy

To run the program, simply download the following files:

1. MNIST_Regression.py
2. main.py

The MNIST_Regression.py program contains the entire Neural Architecture, which is defined inside of a class. To customize your Neural Network, simply create an instance of the MNIST_Regression class with your own parameters. When you type 'python3 main.py' into the terminal window, the Neural Network will train itself, and you will be able to see its performance after its done training via a matplotlib window that pops up. Have fun! 

Heres the documentation for object instantiation:
AI_model = MNIST_Regression(self, learning_ratio=0.1, validation_end=10000, training_start=10000, hidden_layers=[18, 18], num_epochs=500, epoch_step=10);

If you have any questions, email massimoginella12@gmail.com.