import numpy as np
import torch
from torch import nn
import torchvision
from torch.nn import functional as F
torch.set_default_dtype(torch.float64)

from constants import *
class Model():
    """
    Class containing the model and the function to invert gradients to initial inputs
    """
    def __init__(self, pytorch_model, invert_gradient, setup, flatten = True):
        self.setup = setup
        self.pytorch_model = pytorch_model.to(**self.setup).double()
        self.invert_gradient = invert_gradient
        self.loss_fn = torch.nn.CrossEntropyLoss(weight=None, size_average=None, ignore_index=-100,
                                                            reduce=None, reduction='mean')
        n_params = 0
        for param in self.pytorch_model.parameters():
            n_params+=1
        self.n_params = n_params
        self.flatten = flatten

    def __call__(self, x):
        return self.pytorch_model(x)
        
    def parameters(self):
        return self.pytorch_model.parameters()

    def step(self, lr, ground_truth, label):
        """
        Execute one gradient step 
        """
        self.pytorch_model.eval()
        self.pytorch_model.zero_grad()
        if self.flatten:
            ground_truth = ground_truth.flatten().unsqueeze(0)
        target_loss= self.loss_fn(self.pytorch_model(ground_truth), label)
        input_gradient = torch.autograd.grad(target_loss, self.pytorch_model.parameters())
        input_gradient = [grad.detach() for grad in input_gradient]
        with torch.no_grad():
            for j, param in enumerate(self.pytorch_model.parameters()):
                param.copy_(param - lr * input_gradient[j])
        return input_gradient
    
    def update(self, new_params):
        """
        Update the model parameters manually (for initialization or aggregation)
        """
       
        with torch.no_grad():
            for j, param in enumerate(self.pytorch_model.parameters()):
                param.copy_(new_params[j])
        
    def deflatten(self, t):
        """
        Function to recover the original parameter configuration from the flattened tensor t
        """
        param = list(self.pytorch_model.parameters())
        n_params = len(param)
        assert len(torch.cat([param[j].flatten() for j in range(n_params)])) == len(t), f"Length of flattened tensor {len(t)} does not match the number of parameters {len(torch.cat([param[j].flatten() for j in range(n_params)]))}"
        res = []
        i = 0
        for j in range(n_params):
            d = len(param[j].flatten())
            res.append(t[i:i+d].reshape(param[j].shape))
            i+=d
        return res

    def invert(self, inp, lr, *args):
        """
        Function to invert the received updates to recover initial datapoint in correct shape
        """
        
        input_gradient = self.deflatten(-1/lr*inp) 
        input_gradient = [torch.tensor(el , **self.setup) for el in input_gradient]
        return self.invert_gradient(input_gradient, *args) # args[0], args[1], args[2])
        

## Some examples of model implementations 

# Fully connected layer model
class FC_Model(nn.Module):
    def __init__(self, input_dim, output):
        super(FC_Model, self).__init__()
        self.seq = nn.Sequential(
                   nn.Linear(input_dim,output)
                   )

    def forward(self, x):
        return self.seq(x)

def invert_gradient_FC_Model(gr, *args):
    # At least one gradient with respect to the bias should not be null
    if (gr[1]==0).all():
        raise Exception("Bias gradient is null")
    idx = np.argmax(np.abs(gr[1]))
    output = 1/gr[1][idx] * gr[0][idx]

    return output.reshape(1,3,32,32)



class CNN(torch.nn.Module):
    """ Simple, small CNN model.
    """

    def __init__(self, nb_classes):
        """ Model parameter constructor.
        """
        super().__init__()
        # Build parameters
        self.conv1 = nn.Conv2d(1, 32, 5, 1)
        self.conv2 = nn.Conv2d(32, 64, 5, 1)

        self.fc1 = nn.Linear(4*4*64, 1024)
        self.fc2 = nn.Linear(1024, nb_classes)

    def forward(self, x):
        # Forward pass
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))

        x = F.max_pool2d(x, 2, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x