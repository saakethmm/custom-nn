import numpy as np
from keras.datasets import cifar10
from keras.utils import np_utils
#import cv2


import LossFunction
from Network import Network
from FCLayer import FCLayer
from ActivationFunction import ActivationLayer
from LossFunction import mse, mse_prime

def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    #X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
    #X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])
    X_train = X_train.reshape(X_train.shape[0], 1, 3*32*32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], 1, 3*32*32) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    network = Network(mse, mse_prime)
    network.add_layer(FCLayer(32*32*3, 32*16))
    network.add_layer(ActivationLayer('tan'))
    network.add_layer(FCLayer(32*16, 64))
    network.add_layer(ActivationLayer('tan'))
    network.add_layer(FCLayer(64, 10))

    network.train(X_train[0:1000], y_train[0:1000], niter=3, lr=0.1)
    num_test = 100
    results = (network.predict(X_test[0:num_test]))
    results2, acc = network.accuracy(results, y_test[0:num_test])
    print(acc)


if __name__ == "__main__":
    main()
