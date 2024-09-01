import torch
import torch.nn as nn
import numpy as np

def softmax(x):#in numpy
    return (np.exp(x))/ np.sum(np.exp(x), axis = 0)

x = np.array([2.0, 1.0, 0.1])
ouputs = softmax(x)

# in pytorch

x = torch.tensor([2.0, 1.0, 0.1])
outputs = torch.softmax(x, dim = 0)

#### Cross entropy loss######

loss = nn.CrossEntropyLoss()
## This already applies softmax

Y = torch.tensor([2, 0, 1]) # should NOT be one hot encoded
Y_pred_good = torch.tensor([0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1])
Y_pred_bad = torch.tensor([2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1])

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)


###### Multi layer neural net ######

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLu()
        self.linear2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)

        return out

model = NeuralNet(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss()

# For a binary classification problem, we need to add a sigmoid at the ed and then us nn.BCELoss()