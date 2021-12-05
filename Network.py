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

    def maxOutput(self, input):
        results = []
        for result in input:
            zeros = np.zeros(result.shape)
            max_index = np.argmax(result)
            zeros[0][max_index] = 1
            results.append(zeros)
        return results

    def accuracy(self, pred, truth):
        sum = 0
        for i in range(len(pred)):
            if np.array_equal(pred[i], truth[i]):
                sum = sum + 1
        return sum/len(pred)




