####### its so much easier with lightning 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import Trainer


input_size = 784 #28x28 pixel image
hidden_size = 100
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01



class NeuralNet(pl.LightningModule):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out

    def training_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, labels)

        tensorboard_logs = {'train_loss':loss}
        return {'loss' : loss, 'log':tensorboard_logs}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = learning_rate)

    def train_dataloader(self):
        train_dataset = torchvision.datasets.MNIST(root='./data', train = True, download = True, transform = transforms.ToTensor())
        train_loader = DataLoader(train_dataset, batch_size = batch_size,num_workers = 8, shuffle = True)
        return train_loader

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        images = images.reshape(-1, 28*28)

        y_hat = self(images)
        loss = F.cross_entropy(y_hat, labels)
        return {'val_loss' : loss}

    

    def val_dataloader(self):
        val_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor())
        val_loader = DataLoader(val_dataset, batch_size = batch_size,num_workers = 8, shuffle = False)
        return val_loader



trainer = Trainer(max_epochs = num_epochs, fast_dev_run=False) # runs one epoch just to see if model works if fast = true
model = NeuralNet(input_size, hidden_size, num_classes)
trainer.fit(model)



