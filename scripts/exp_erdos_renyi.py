import numpy as np
from scipy.linalg import eigh
import networkx as nx
from matplotlib import pyplot as plt
import itertools
from tqdm import tqdm, trange
import time
import pickle 
import os 
import json
import string
import random
import sys

sys.path.append("src")

from utils import *
from consensus import Reconstruct

def run_exp(n_list, p_list, attackers, n_exp, full_reconstruct=False):
    """
    p_list is the list of p's used for the first n in n_list
    for other values of n, we multiply p_list by a factor to ensure that n*p_list is constant
    """
    res = np.zeros((len(n_list), len(p_list), n_exp))  
    for n_idx, n in enumerate(n_list):
        for p_idx, p in enumerate(p_list):
            invertible_graphs = 0
            exp_count = 0
            while exp_count < n_exp:
                G = nx.fast_gnp_random_graph(n, n_list[0]*p / n)
                if nx.is_connected(G):
                    R = Reconstruct(G, int(min(n*p/len(attackers),n/2)), attackers)
                    if full_reconstruct:
                        invertible_graphs += int(R.rank() == n)
                        exp_count += 1

                    else:
                        try:
                            U, x, x_hat = R.reconstruction()
                            invertible_graphs += np.mean(np.where(np.isnan(x_hat), 0, 1))
                            exp_count += 1
                            res[n_idx, p_idx, exp_count] = np.mean(np.where(np.isnan(x_hat), 0, 1))
                        except KeyboardInterrupt:
                            raise 
                        except:
                            print("invertion failed")
                            continue
                        


    return res


if __name__ == '__main__':
    # 1 attacker
    n_exp = 20
    n_list = [20,40,60,80,100]
    #n_list = [20,40,60,80,100]
    p_list =  [0.15,0.2,0.25,0.30]
    #p_list = np.array([0.04, 0.06, 0.08, 0.1 ])
    attackers = [0]
    full_reconstruct = False
    res11 = run_exp(n_list, p_list, attackers, n_exp, full_reconstruct )

    print("Experiment finished")
    # 2 attackers
    attackers = [0,1]
    res22 = run_exp(n_list, p_list, attackers, n_exp, full_reconstruct)
    print("Experiment finished")

    # 3 attackers
    #p_list = np.array([0.08, 0.11, 0.14, 0.17 ])
    attackers = [0,1,2]
    res33 = run_exp(n_list, p_list, attackers, n_exp, full_reconstruct )
    print("Experiment finished, plotting .. ")
    

    # Assume res11, res22, res33 are the results from run_exp for 1, 2, and 3 attackers respectively
    # Plotting
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    x_ticks = np.array(p_list) * n_list[0]

    # Plot for each attacker configuration
    for a_idx, res_array in enumerate([res11, res22, res33]):
        for n_idx, n in enumerate(n_list):
            mean_vals = np.mean(res_array[n_idx, :], axis=1)
            std_vals = np.std(res_array[n_idx, :], axis=1)
            ax[a_idx].errorbar(x_ticks, mean_vals, yerr=std_vals, fmt="s--", label=rf'$n={n}$')

    for i, a in enumerate(ax):
        a.legend()
        a.set_xlabel(r'$n \times p$')
        a.set_ylim([0, 1.01])


    ax[0].set_title("1 attacker")
    ax[1].set_title("2 attackers")
    ax[2].set_title("3 attackers")


    exp_dir = f"experiments/erdos_renyi_{full_reconstruct}_{n_list[0]}-{n_list[-1]}"
    
    if os.path.isdir(exp_dir):
        exp_dir = exp_dir +'_' + random.choice(string.ascii_letters)
    if os.path.isdir(exp_dir):
        exp_dir = exp_dir + '_' + random.choice(string.ascii_letters)
    os.makedirs(exp_dir, exist_ok=True)
    print("Created new folder : ", exp_dir)

    plt.savefig(os.path.join(exp_dir, "curves.pdf")) 

    ress = [res11, res22, res33]
    with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
        pickle.dump(ress, f)

    config = {'n_exp' : n_exp ,  'n_list':  n_list, 'p_list' : list(p_list), full_reconstruct : full_reconstruct }
    #json_config = json.dumps(config)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f)
