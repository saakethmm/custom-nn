import numpy as np
from tqdm import tqdm


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
                # sample j
                output = x_train[j]

                # forward propagation
                for layer in self.layers:
                    output = layer.forward_prop(output)

                # loss calculation
                loss = loss + self.loss(y_train[j], output)

                # backward propagation
                error = self.loss_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.backward_prop(error, lr)

            average_loss = loss/len(x_train)
            print('epoch %d/%d   error=%f' % (i+1, niter, average_loss))

    def predict(self, input):
        results = []

        # iterate through each input
        for i in range(len(input)):
            output = input[i]

            # run forward propagation
            for layer in self.layers:
                output = layer.forward_prop(output)

            results.append(output)

        return results

    def maxOutput(self, results):
        return np.argmax(results, axis=1)


