import numpy as np
from Layer import Layer


class ActivationLayer(Layer):
    def __init__(self, type):
        # super(ActivationFunction, self).__init__()
        self.type = type

    def forward_prop(self, input_data):
        self.input = input_data
        self.output = np.zeros(self.input.shape)
        if self.type == "tan":
            self.output = self.tanh(self.input)
        elif self.type == "sigmoid":
            self.output = self.sigmoid(self.input)
        elif self.type == "relu":
            self.output = self.ReLU(self.input)
        elif self.type == "leaky":
            self.output = self.leaky_ReLU(self.input)
        elif self.type == 'softmax':
            self.output = self.softmax(self.input)
        return self.output

    def backward_prop(self, output_error, lr): # TODO: Fix other dimensions
        if self.type == "tan":
            data = self.tanh_prime(self.input)*output_error
            return data
        elif self.type == "sigmoid":
            return self.tanh_prime(self.input)*output_error
        elif self.type == "relu":
            data = self.ReLU_prime(self.input)*output_error
            return data
        elif self.type == "leaky":
            return self.leaky_ReLU_prime(self.input)*output_error
        elif self.type == "softmax":
            return self.softmax_prime(self.input)*output_error

    def tanh(self, x):
        return np.tanh(x)

    def tanh_prime(self, x):
        data = 1-np.tanh(x)**2
        return data

    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    def sigmoid_prime(self, x):
        return np.exp(-x)/(1 + np.exp(-x))**2

    def ReLU(self, x):
        return x * (x > 0)

    def ReLU_prime(self, x):
        data = 1. * (x > 0)
        return data

    def leaky_ReLU(self, x):
        alpha = 0.1
        return np.maximum(alpha*x, x)

    def leaky_ReLU_prime(self, x):
        alpha = 0.1
        return 1. * (x >= 0) + alpha * (x < 0)

    def softmax(self, x):
        st_x = np.exp(x - np.max(x))
        return st_x/st_x.sum(axis=0)

    def softmax_prime(self, x):
        pass
