import numpy as np
from Layer import Layer

# Single fully connected layer
class FCLayer (Layer):
    def __init__(self, input_size, output_size):
        # Initializes the Layer class
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size, 1)

    # Returns output for a given input
    def forward_prop(self, input):
        self.input = input
        self.output = np.matmul(self.weights, self.input) + self.bias
        return self.output

    # Given dE/dY from activation layer function?
    def backward_prop(self, output_error, learning_rate):
        """
        Returns the error for the input layer (output layer of previous layer)\n
        Updates the weight and bias errors given the output error\n
        Calls gradient_desc to perform gradient descent on the parameters
        """
        self.weight_error = np.matmul(self.input, output_error.T)
        self.bias_error = self.output_error # TODO: May want to remove to save space ...
        self.input_error = np.matmul(self.weights, output_error)

        self.update_weights(learning_rate)

        return self.input_error

    def update_weights(self, a):
        self.weights -= a * self.weight_error
        self.bias -= a * self.bias_error

