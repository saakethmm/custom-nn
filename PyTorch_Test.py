import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import sys

import math
from collections import OrderedDict

class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            #nn.Conv2d(3, 8, kernel_size=3, padding=1),
            #nn.MaxPool2d(2),
        )
        self.fc = nn.Sequential(
            # TODO: fully-connected layer (64->64)
            nn.Linear(32*32*3, 64),

            # TODO: ReLU
            nn.ReLU(),

            # here you can try adding more fully-connected layers followed by
            # ReLU, if you want.

            # TODO: fully-connected layer (64->10)
            nn.Linear(64, 10)

            # the softmax will be part of the cross entropy loss (defined
            # in main()) so we just need to have a linear layer with output size
            # equal to the number of classes (10). This is what is accomplished
            # by the layer you will implement above.

        )

    def forward(self, x):
        x = self.conv(x)
        # if you decide to change or add anything to conv(), you will need to
        # change x.view(-1, num_feats) where num_feats is the number of scalar
        # output features from conv(). You will then need to change the first
        # input layer in fc() to be num_feats as well.l
        x = x.view(-1, 32*32*3)
        x = self.fc(x)
        return x


def train(trainloader, net, criterion, optimizer, device):
    for epoch in range(20):  # loop over the dataset multiple times
        start = time.time()
        running_loss = 0.0
        for i, (images, labels) in enumerate(trainloader):
            images = images.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward pass
            yhat = net.forward(images)
            loss = criterion(yhat, labels)
            # backward pass
            loss.backward()
            # optimize the network
            optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % 100 == 99:  # print every 2000 mini-batches
                end = time.time()
                print('[epoch %d, iter %5d] loss: %.3f eplased time %.3f' %
                      (epoch + 1, i + 1, running_loss / 100, end - start))
                start = time.time()
                running_loss = 0.0
    print('Finished Training')


def test(testloader, net, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))


def main(dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root=dir, train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True)

    testset = torchvision.datasets.CIFAR10(root=dir, train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100,
                                             shuffle=False)
    net = VGG().to(device)
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    train(trainloader, net, criterion, optimizer, device)
    test(testloader, net, device)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <directory to CIFAR 10 data folder>")
    else:
        dir = sys.argv[1]
        main(dir)

