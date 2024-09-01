import torch

x = torch.randn(3, requires_grad = True)
print(x)

y = x+2  # forward pass of the model
print(y)

z = y*y*2
z = z.mean()
print(z)

z.backward() #dz/dx

print(x.grad)

# x.require_grad_(False)
# x.detach()
# with torch.no_grad():  These 3 help remove the gradient

#After every iteration while training the model, we need to initialize grads back to zero by doing
# weights.grad.zero_()

weights = torch.ones(4, requires_grad = True)

optimizer = torch.optim.SGD(weights, lr = 0.01)
optimizer.step()
optimizer.zero_grad()