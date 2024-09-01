import numpy as np
import torch
import torch.nn as nn


######## without pytorch ########
X = np.array([1, 2, 3, 4], dtype = np.float32)
Y = np.array([2, 4, 6, 8], dtype = np.float32)

w = 0.0

def forward(x):
    return w*x

def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()

def gradient(x, y, y_predicted):
    return np.dot(2*x, y_predicted - y).mean()

learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):

    y_pred = forward(X)

    l = loss(Y, y_pred)

    dw = gradient(X, Y, y_pred)

    w -= learning_rate*dw

########## with pytorch #########

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w*x

def loss(y, y_predicted):
    return ((y_predicted - y)**2).mean()



learning_rate = 0.01
n_iters = 10

for epoch in range(n_iters):

    y_pred = forward(X)

    l = loss(Y, y_pred)

    l.backward()

    with torch.no_grad():
        w -= learning_rate*w.grad

    w.grad.zero_()

###### without loss and parameter updates #########

X = torch.tensor([1, 2, 3, 4], dtype = torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype = torch.float32)

w = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)

def forward(x):
    return w*x

learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD([w], lr = learning_rate)



for epoch in range(n_iters):

    y_pred = forward(X)

    l = loss(Y, y_pred)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

########## without forward #########

X = torch.tensor([[1], [2], [3], [4]], dtype = torch.float32)
Y = torch.tensor([[2], [4], [6], [8]], dtype = torch.float32)

n_samples, n_features = X.shape

input_size = n_features
output_size = n_features

model = nn.Linear(input_size, output_size)

learning_rate = 0.01
n_iters = 10

loss = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters, lr = learning_rate)



for epoch in range(n_iters):

    y_pred = model(X)

    l = loss(Y, y_pred)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()