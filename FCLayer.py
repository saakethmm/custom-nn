import numpy as np
from Layer import Layer

# Single fully connected layer
class FCLayer (Layer):
    def __init__(self, input_size, output_size):
        # Initializes the Layer class
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

    # Returns output for a given input
    def forward_prop(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    # Given dE/dY from activation layer function?
    def backward_prop(self, output_error, learning_rate):
        """
        Returns the error for the input layer (output layer of previous layer)\n
        Updates the weight and bias errors given the output error\n
        Calls gradient_desc to perform gradient descent on the parameters
        """
        self.input_error = np.dot(output_error, self.weights.T)
        self.weight_error = np.dot(self.input.T, output_error)
        self.bias_error = output_error # TODO: May want to remove to save space ...

        self.update_weights(learning_rate)

        return self.input_error

    def update_weights(self, a):
        self.weights -= a * self.weight_error
        self.bias -= a * self.bias_error

