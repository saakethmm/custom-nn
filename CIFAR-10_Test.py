import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys
from keras.datasets import cifar10
from keras.utils import np_utils

import LossFunction
from Network import Network
from FCLayer import FCLayer
from ActivationFunction import ActivationLayer
from LossFunction import mse, mse_prime

def main():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.reshape(X_train.shape[0], 1, 3*32*32) / 255.0
    X_test = X_test.reshape(X_test.shape[0], 1, 3*32*32) / 255.0
    y_train = np.eye(10)[y_train]
    y_test = np.eye(10)[y_test]

    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    network = Network(mse, mse_prime)
    network.add_layer(FCLayer(32*32*3, 64))
    network.add_layer(ActivationLayer('relu'))
    network.add_layer(FCLayer(64, 10))

    network.train(X_train[0:1000], y_train[0:1000], niter=100, lr=0.1)


    '''
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), transforms.Grayscale(num_output_channels=1)])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)
    print(len(trainset))
    print(len(trainset[0]))
    print(len(trainset[0][0]))
    print(len(trainset[0][0][0]))
    print(len(trainset[0][0][0][0]))

    #print(trainset[0][0][0][0])
    '''


if __name__ == "__main__":
    main()