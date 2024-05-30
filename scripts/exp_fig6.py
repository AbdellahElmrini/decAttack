


import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import warnings
warnings.filterwarnings("ignore")
import os 
import sys
import pickle
sys.path.append("src")

from decentralized import *
from graphs import *
from models import *
from utils import *
from metrics import *
import matplotlib 
import torch
from tqdm import tqdm, trange
matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42
import seaborn as sns
sns.set_theme()


# Defining the attack parameters
setup = system_startup()
n = 35
G = line_graph(n)
N = G.number_of_nodes()
n_iter = 35
attackers = [0]
W = LaplacianGossipMatrix(G)
Wt = torch.tensor(W).float()
R = ReconstructOptim(G, n_iter, attackers)


torch.manual_seed(42)
model = Model(FC_Model(3072, 10), invert_gradient_FC_Model, setup)



n_exp = 30
np.random.seed(0)
seeds = np.random.randint(0, 10000, n_exp)
psnrs = []
sqs = []
rss = []
for seed in tqdm(seeds):
    D = Decentralized(R, model, setup, seed = seed, targets_only = True)
    target_inputs = torch.cat(D.data)[1:] 
    lr = 1e-5
    D.run(lr);
    x_hat = R.reconstruct_LS_target_only(D.sent_params, D.attackers_params)
    outputs = []
    for j in range(n-1):
        outputs.append(model.invert( x_hat[j], lr) ) 
    psnr_score, sq_score, relative_score = compute_metrics(torch.cat(outputs),target_inputs)
    psnrs.append(psnr_score)
    sqs.append(sq_score)
    rss.append(relative_score)
    

fig, ax1 = plt.subplots()

color = 'tab:red'
ax1.set_xlabel('Distance to the attacker')
ax1.set_ylabel('PSNR', color=color)
ax1.errorbar( range(1,n-1), torch.stack(psnrs).mean(0)[1:], yerr = torch.stack(psnrs).std(0)[1:],fmt= 'o-', color = color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:blue'
ax2.set_ylabel('Relative Square Loss', color=color)  
ax2.errorbar( range(1,n-1), torch.tensor(rss).mean(0)[:-1], yerr = torch.tensor(rss).std(0)[:-1],fmt= 'o-', color = color)

ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()  
plt.savefig("outputs/gd/line_graph_psnr_rss.pdf", format="pdf")
plt.show()
