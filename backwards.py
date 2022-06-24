import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from numpy import array
from numpy import argmax
import tensorflow as tf
import numpy as np
import numpy as np

from tensorflow.keras.utils import to_categorical

device = torch.device('cpu')

input_size = 784
hidden_size = 10
output_size = 10
num_epochs = 5

train_dataset = torchvision.datasets.MNIST(root='MNISTdata', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = torchvision.datasets.MNIST(root='MNISTdata', train=False, transform=transforms.ToTensor())
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=50, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1)

for inputs, targets in train_loader:
    f=nn.Flatten()
    inputs=f(inputs) #.to(device)
    #targets = targets.to(device)
    break


class NN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 500, bias=False)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(500, 10, bias=False)

    def forward(self, x):
        a1 = self.fc1(x)
        a2 = self.relu(a1)
        out = self.fc2(a2)

        return a1,a2, out

    def weight(self):
        w2 = self.fc2.weight
        w1 = self.fc1.weight
        return w1, w2



model = NN(input_size, hidden_size, output_size).to(device)

encoded = to_categorical(targets,num_classes=10)
encoded = torch.from_numpy(encoded)
a1,a2, outputs = model.forward(inputs)
w1, w2 = model.weight()

lossfunc = nn.CrossEntropyLoss(reduction = 'sum')
loss = lossfunc(outputs, targets)
loss.backward()
#torch.mean(outputs).backward()
w2_grad = torch.detach(w2._grad)
w1_grad = torch.detach(w1._grad)
print(torch.detach(w2._grad))
softmaxfunc = nn.Softmax(dim = 1)
y_head = softmaxfunc(outputs)
w2 = torch.transpose(w2, 0, 1)
error3 = y_head - encoded
error3 = torch.transpose(error3, 0, 1)

d1 = torch.matmul(w2, error3)
#d1 = torch.transpose(d1, 0, 1)
#g_z2 = a2 * (1 - a2)
g_z2 = (a2 > 0) * 1
g_z2 = torch.transpose(g_z2, 0, 1)
g_z2 = g_z2 * d1


d1 = torch.matmul(g_z2, inputs)

d2 = torch.matmul(error3, a2)
print (d2)

#print(torch.equal(d2,w2.grad))
print (d2  - w2_grad)

print(torch.all(torch.lt(torch.abs(torch.add(d2, -w2_grad)), 1e-6)))

print(torch.all(torch.lt(torch.abs(torch.add(d1, -w1_grad)), 1e-6)))

