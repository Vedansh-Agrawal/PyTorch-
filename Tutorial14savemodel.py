import torch
import torch.nn as nn

## Following is how to save your model:
#torch.save(model, PATH)
#Then load model with
#model = torch.load(PATH)
#model.eval()


###Recomended method
# Only save models parameters
# torch.save(model.state_dict(), PATH)

#load model
# model = Model(*args, **kwargs)
#model.load_state_dict(torch.load(PATH)
#model.eval()


####EXAMPLE########

class Model(nn.Module):
    def __init__(self, n_input_features):
        super(Model, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)

    def forward(self, x):
        y_pred = torch.sigmoid(self.linear(x))
        return y_pred

#model = Model(n_input_features = 6)

learning_rate = 0.001
#optimizer = torch.optim.SGD(model.parameters, lr = learning_rate)

## make checkpoint

#checkpoint = {
#    "epoch" : 90,
#    "model_state": model.state_dict(),
#    "optim_state": optimizer.state_dict()
#}


FILE = 'model.pth'

#torch.save(checkpoint, FILE)
loaded_checkpoint = torch.load(FILE)
epoch = loaded_checkpoint['epoch']

model = Model(n_input_features=6)
optimizer = torch.optim.SGD(model.parameters(), lr=0)

model.load_state_dict(loaded_checkpoint['model_state'])
optimizer.load_state_dict(loaded_checkpoint['optim_state'])


#torch.save(model.state_dict(), FILE)

#model = torch.load(FILE)
#model.eval()

#loaded_model = Model(n_input_features = 6)
#loaded_model.load_state_dict(torch.load(FILE))
#loaded_model.eval()

#for param in loaded_model.parameters():
#    print(param)
