import numpy as np


class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    # Does this need to be a base class function?
    def update_weights(self, lr, bs):
        raise NotImplementedError

    def forward_prop(self, input):
        return NotImplementedError

    def backward_prop(self, output_error, lr):
        return NotImplementedError
