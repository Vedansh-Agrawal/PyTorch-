import torch
import numpy as np

########making tensors#######

x = torch.empty(2, 3)  # Stores empty tensor with shape

y = torch.rand(2, 2) # stores random values

z = torch.ones(2, 2) # stores 1

a = torch.zeros(2, 4, 3, dtype = torch.float16) # stores 0 of data type float

print(x.size()) #prints size

b = torch.tensor([2.5, 0.1]) # initializing tensors

############################
#######arithemetic##########

x = torch.rand(2, 2)
y = torch.rand(2, 2)

z = x+y # element wise addition
z = torch.add(x, y) #does the same thing

y.add_(x) # adding underscore allows for inplace operation by concatenating to a list


z = x - y # element wise subtraction
z = torch.sub(x, y)

z = x*y
z = torch.mul(x, y)

z = x / y
z = torch.div(x, y)

####################################
########slicing and reshaping#######

x = torch.rand(4, 4)
y = x[:, 0] #only column 0
z = x[0, :] # only row 0

y = x.view(8, -1) # reshapes to size given, here it is 8, -1 where the -1 acts as an autofill to the shape



a = np.ones(5)
b = torch.from_numpy(a) # converts to tensor default dtype is float64

x = torch.ones(5, requires_grad = True) #this tells pytorch that we need to get the gradient for the tensor