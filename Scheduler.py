import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler

lr = 0.1

model = nn.Linear(10, 1)

optimizer = torch.optim.Adam(model.parameters(), lr = lr)

# Multiplies function to lr
#lambda1 = lambda epoch: epoch/10
#scheduler = lr_scheduler.LambdaLR(optimizer, lambda1)

# Mulitplies scalar to lr
#lambda1 = lambda epoch: 0.95
#scheduler = lr_scheduler.MultiplicativeLR(optimizer, lambda1)

# Multiplies scalar to lr after n steps
scheduler = lr_scheduler.StepLR(optimizer, step_size = 30, gamma = 0.1)

for epoch in range(5):
    optimizer.step()

    scheduler.step()
    print(optimizer.state_dict()['param_groups'][0]['lr'])