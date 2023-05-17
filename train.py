#!/usr/bin/python

import pandas as pd
import numpy as np
import zadanie1 as z
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sys

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #self.conv1 = nn.Conv2d(3, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        #self.fc1 = nn.Linear(16 * 5 * 5, 120)
        #self.fc2 = nn.Linear(20, 6)
        self.fc3 = nn.Linear(6, 6)

    def forward(self, x):
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = torch.flatten(x, 1)
        #x = F.relu(self.fc1(x))
        #x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def trainNet(trainloader, criterion, optimizer, epochs=20):
    for epoch in range(epochs):

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            labelsX = torch.Tensor([x for x in labels])
            labels = labelsX.type(torch.LongTensor)

            optimizer.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    print('Finished Training')
    
if __name__ == '__main__':

    train, dev, test = z.prepareData()

    batch_size = 4

    trainlist = train.values.tolist()
    testlist = test.values.tolist()

    trainset = [[torch.Tensor(x[1:]), torch.Tensor([x[0]])] for x in trainlist]
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = [[torch.Tensor(x[1:]), torch.Tensor([x[0]])] for x in testlist]
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('male', 'female')

    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    epochs = int(sys.argv[1])
    print(epochs)

    trainNet(trainloader, criterion, optimizer, int(float(epochs)))

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)

