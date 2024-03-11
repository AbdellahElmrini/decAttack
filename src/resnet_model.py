
import numpy as np
import torch
from torch import nn
import torchvision

import sys 
#import pathlib
#path_inv = str(pathlib.Path().absolute()) + "/invertinggradients"
sys.path.append("src")
sys.path.append("invertinggradients")
import inversefed
from inversefed import consts
from models import Model
from utils import *
import warnings
warnings.filterwarnings('ignore')


# Resnet18

def invert_gradient_resnet18(gr, label, model, config) :
    setup = system_startup()
    dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(gr, label, img_shape=(3, 32, 32))
    output = output.reshape(1,3,32,32)
    return output, stats


def invert_gradient_MNIST(gr, label, model, config) :
    setup = system_startup()
    dm = torch.as_tensor((0.1307,), **setup)[:, None, None]
    ds = torch.as_tensor((0.3081,), **setup)[:, None, None]
    rec_machine = inversefed.GradientReconstructor(model, (dm, ds), config, num_images=1)
    output, stats = rec_machine.reconstruct(gr, label, img_shape=(1, 28, 28))
    output = output.reshape(1,1,28,28)
    return output, stats