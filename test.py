import pandas as pd
import numpy as np
import zadanie1 as z
import train as tr
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

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

testdata = []

def testNet(testloader):

    PATH = './cifar_net.pth'
    net = Net()
    net.load_state_dict(torch.load(PATH))

    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            input, labels = data
            
            labelsX = torch.Tensor([x for x in labels])
            labels = labelsX.type(torch.LongTensor)

            outputs = net(input)

            _, predicted = torch.max(outputs.data, 1)
            testdata.append([input, labels, predicted])
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    #print(f'Accuracy of the network: {100 * correct // total} %')

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

    testNet(testloader)

    with open('testresults.txt', 'w') as the_file:
        for item in testdata:
            for i in range(len(item)):
                the_file.write(f'data: {item[0][i]} \n true value: {item[1][i]} \n prediction: {item[2][i]}\n')
                print(f'data: {item[0][i]} \n true value: {item[1][i]} \n prediction: {item[2][i]}\n')

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for item in testdata:
        for i in range(len(item)):
            if int(item[1][i]) and int(item[2][i]):
                tp += 1
            elif not int(item[1][i]) and not int(item[2][i]):
                tn += 1
            elif not int(item[1][i]) and int(item[2][i]):
                fp += 1
            elif int(item[1][i]) and not int(item[2][i]):
                fn += 1

    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fscore = (2 * precision * recall) / (precision + recall)

    with open('metrics.txt', 'w') as the_file:
        the_file.write(f'Accuracy: {accuracy} \nPrecision: {precision} \nRecall: {recall} \nF-score: {fscore} \n')
    