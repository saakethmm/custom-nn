import numpy as np


class Network:
    def __init__(self, loss, loss_prime):
        self.layers = []
        self.loss = loss
        self.loss_prime = loss_prime

    def add_layer(self, layer):
        self.layers.append(layer)

    def train(self, x_train, y_train, niter, lr):
        for i in range(niter):
            loss = 0
            for j in range(len(x_train)):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_prop(output)

                loss = loss + self.loss(y_train[j], output)

                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backProp(error, lr)

            average_loss = loss/len(x_train)
            print('epoch %d/%d   error=%f' % (i+1, niter, average_loss))

    def predict(self, input):
        results = []

        for i in range(len(input)):
            output = input[i]
            for layer in self.layers:
                output = layer.forward_prop(output)
            results.append(output)

        return results
