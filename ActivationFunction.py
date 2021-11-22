import numpy as np

class ActivationFunction:
    def __init__(self, type):
        super(ActivationFunction, self).__init__()
        self.type = None

    def forwardPop(self):
        return 0

    def backProp(self):
        return 0

    def tanh(self):
        return 0