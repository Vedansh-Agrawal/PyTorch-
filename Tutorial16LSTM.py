import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import sys
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#input_size = 784 #28x28 pixel image
hidden_size = 128
num_classes = 10
num_epochs = 2
batch_size = 100
learning_rate = 0.01

input_size = 28
sequence_length = 28
num_layers = 2

train_dataset = torchvision.datasets.MNIST(root='./data', train = True, transform = transforms.ToTensor(), download = True)
test_dataset = torchvision.datasets.MNIST(root='./data', train = False, transform = transforms.ToTensor())

train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)


examples = next(iter(test_loader))
samples, labels  = examples

for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(samples[i][0], cmap = 'gray')
#plt.show()
img_grid = torchvision.utils.make_grid(samples)
writer.add_image('mnist_images', img_grid)
writer.close()


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers):
        super(RNN, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first = True)

        self.fc = nn.Linear(hidden_size, num_classes)

        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), lr = learning_rate)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)

        return out

model = RNN(input_size, hidden_size, num_classes, num_layers)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

writer.add_graph(model, samples.reshape(-1, 28*28))
writer.close()

n_total_steps = len(train_loader)

running_loss = 0.0
running_correct = 0.0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):

        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        
        running_loss += loss.item()

        _, predictions = torch.max(outputs, 1)
        running_correct += (predictions == labels).sum().item()

        if (i + 1)%100 == 0:
            print(f'epoch{epoch+1}/{num_epochs}, step {i+1} / {n_total_steps}, loss = {loss.item():.4f}')
            writer.add_scalar('training loss', running_loss / 100, epoch * n_total_steps + i)
            writer.add_scalar('accuracy', running_correct / 100, epoch * n_total_steps + i)
            running_loss = 0.0
            running_correct = 0.0

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        outputs = model(images)

        _, predictions = torch.max(outputs, 1)
        n_samples += labels.shape[0]
        n_correct += (predictions == labels).sum().item()
    
    acc =  100.0 * n_correct/n_samples
    print(f'accuracy = {acc}')