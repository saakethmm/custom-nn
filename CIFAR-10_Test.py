import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import time
import sys

from Network import Network
from FCLayer import FCLayer
from ActivationFunction import ActivationLayer
from LossFunction import mse, mse_prime

def main():
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


if __name__ == "__main__":
    main()