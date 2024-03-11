import matplotlib.pyplot as plt
import numpy as np
import networkx as ns
import warnings
warnings.filterwarnings("ignore")

import sys
sys.path.append("src")

from decentralized import *
from graphs import *
from models import *
from utils import *
from metrics import *
import torch

# Defining the attack parameters
setup = system_startup()
n = 31
G = line_graph(n)
N = G.number_of_nodes()
n_iter = 31
attackers = [0]
W = LaplacianGossipMatrix(G)
Wt = torch.tensor(W).float()
R = ReconstructOptim(G, n_iter, attackers)

# Defining the model
torch.manual_seed(0)
model = Model(FC_Model(3072, 10), invert_gradient_FC_Model, setup)
D = Decentralized(R, model, setup, seed = 999, targets_only = True)

lr = 0.0001
D.run(lr);

x_hat = R.reconstruct_LS_target_only(D.sent_params, D.attackers_params)

outputs = []
for j in range(n-1):
    outputs.append(model.invert( x_hat[j], lr))

target_inputs = torch.cat(D.data)[1:] 


plot_tensors(target_inputs, multiline=False, save=True, save_folder="outputs/", suffix="_input", row_size=15)

plot_tensors(torch.cat(outputs), title = 'outputs', multiline=False, save=True, save_folder="outputs/", suffix="_output", row_size=15)