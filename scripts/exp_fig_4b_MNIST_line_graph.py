import matplotlib.pyplot as plt
import numpy as np
import networkx as ns
import warnings
import torch
import sys
sys.path.append("src")
sys.path.append("invertinggradients")
import torchvision
from decentralized import *
from graphs import *
from models import *
from utils import *
from resnet_model import *
from metrics import *
import os
import logging
import random
import string
import time
import pickle
# GPU or CPU 
setup = system_startup()
print(setup)

# Defining the attack parameters
n = 31
G = line_graph(n)
N = G.number_of_nodes()
n_iter = 31
attackers = [0]
W = LaplacianGossipMatrix(G)
Wt = torch.tensor(W).float()

R = ReconstructOptim(G, n_iter, attackers)



# Inversion parameters
config = dict(signed=True,
                    boxed=True,
                    cost_fn='sim',
                    indices='def',
                    weights='equal',
                    lr=0.1,
                    optim='adam',
                    restarts=1,
                    max_iterations=20,
                    total_variation=1e-1,
                    init='randn',
                    filter='none',
                    lr_decay=True,
                    scoring_choice='loss')

# Defining the model

print("Building model")

torch.manual_seed(1)
cnn_model = CNN(10)

model = Model(cnn_model, invert_gradient_MNIST, flatten = False, setup = setup)

# Decomment the next lines to use a resnet18 model instead
#resnet = torchvision.models.resnet18()
#model = Model(resnet, invert_gradient_resnet18, flatten = False, setup = setup)
# Defining and running the decentralized protocol
D = Decentralized(R, model, setup, data_name= "MNIST", targets_only = True)
pretrain_it, pretrain_lr =  0, 0

# Decomment the next line and changee the previous values to pretrain the model.
#grs = D.pretrain(pretrain_it, pretrain_lr)
#print("Pretraining finished")

lr = 1e-6
D.run(lr)


grs= D.gradients
for j in range(6):
    print(torch.linalg.norm(grs[j]))
x_hat = R.reconstruct_LS_target_only(D.sent_params, D.attackers_params)


print('Absolute errors')
print(square_loss( D.gradients[1:N], -1/lr*torch.tensor(x_hat) ))
print('Relative errors')
print(relative_square_loss(D.gradients[1:N] , -1/lr*torch.tensor(x_hat)))



save_folder = f'experiments/gd'

outputs = []
for j in range(N-1): #
    ground_truth = D.data[j+1]
    output, stats = model.invert(x_hat[j], lr, D.labels[j+1], D.models[j+1].pytorch_model, config)
    outputs.append(output)
    test_mse = (output.detach() - ground_truth).pow(2).mean().item()
    feat_mse = (D.models[j+1](output.detach())- D.models[j+1](ground_truth)).pow(2).mean().item()

    test_psnr = psnr(output, ground_truth, factor=1.0, batched = True)

    print(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
          f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");
    logging.info(f"Rec. loss: {stats['opt']:2.4f} | MSE: {test_mse:2.4f} "
            f"| PSNR: {test_psnr:4.2f} | FMSE: {feat_mse:2.4e} |");

    plot_tensors(output, True, save_folder = save_folder, suffix = str(j), data ='mnist')

cpu_outputs=  [output.detach().cpu() for output in outputs]
cpu_inputs= [input.detach().cpu() for input in D.data]
pickle.dump(cpu_outputs, open(save_folder + '/outputs.pkl', 'wb'))
pickle.dump(cpu_inputs, open(save_folder + '/data.pkl', 'wb'))
            
D.plot_inputs(True, save_folder = save_folder, suffix=  "_input", data = "mnist", multiline = False)
plot_tensors(torch.cat(outputs), True, save_folder = save_folder, suffix = '_output', data ='mnist', multiline = False)