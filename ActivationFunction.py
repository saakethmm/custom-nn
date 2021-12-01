import numpy as np
from Layer import Layer


class ActivationLayer(Layer):
    def __init__(self, type):
        Layer.__init__(self)
        # super(ActivationFunction, self).__init__()
        self.type = type

    def forwardPop(self, input_data):
        self.input = input_data
        if type == "tan":
            self.output = self.tanh(self.input)
        elif type == "sigmoid":
            self.output = self.sigmoid(self.input)
        elif type == "relu":
            self.output = self.ReLU(self.input)
        elif type == "leaky":
            self.output = self.leaky_ReLU(self.input)
        return self.output

    def backProp(self, output_error):
        if type == "tan":
            return self.tanh_prime(self.input)*output_error
        elif type == "sigmoid":
            return self.tanh_prime(self.input)*output_error
        elif type == "relu":
            return self.ReLU_prime(self.input)*output_error
        elif type == "leaky":
            return self.leaky_ReLU_prime(self.input)*output_error

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        return 1-np.tanh(x)**2

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x)/(1 + np.exp(-x))**2

    def ReLU(self, x):
        return x * (x > 0)

    def ReLU_prime(self, x):
        return 1. * (x > 0)

    def leaky_ReLU(self, x):
        alpha = 0.1
        return np.max(alpha*x, x)

    def leaky_ReLU_prime(self, x):
        alpha = 0.1
        return 1 if x>=0 else alpha