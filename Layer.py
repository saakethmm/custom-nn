import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forwardPop(self, input):
        return 0

    def backProp(self, output_error):
        return 0