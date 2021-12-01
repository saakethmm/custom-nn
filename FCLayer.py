import numpy as np
import Layer

# Single fully connected layer
class FCLayer (Layer):
    def __init__(self, input_size, output_size):
        # Initializes the Layer class
        self.weights = np.random.rand(input_size, output_size)
        self.bias = np.random.rand(output_size, 1)

    # Returns output for a given input
    def forwardProp(self, input):
        self.input = input
        self.output = np.matmul(self.weights, self.input) + self.bias
        return self.output


    def backwardProp(self):
        return 0