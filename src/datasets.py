import torch
import torchvision
from torchvision import transforms
from constants import *
torch.set_default_dtype(torch.float64)




def build_cifar10(setup, root = "images"):
    trainset = torchvision.datasets.CIFAR10(root=root+"/cifar10", train=True,
                                            download=True, transform = transforms.ToTensor())
    dm = torch.as_tensor(cifar10_mean, **setup)[:, None, None]
    ds = torch.as_tensor(cifar10_std, **setup)[:, None, None]

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(dm, ds)]) 

    trainset.transform = transform

    return trainset

def build_MNIST(setup, root = "images"):
    trainset = torchvision.datasets.MNIST(root=root+"/MNIST" , train=True,
                                            download=True, transform = transforms.ToTensor())

        
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]) 

    trainset.transform = transform

    return trainset


datasets = {"Cifar10": build_cifar10, "MNIST": build_MNIST}