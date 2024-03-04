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
from src.utils import *
from src.consensus import Reconstruct
import logging
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42

matplotlib.rcParams['ps.fonttype'] = 42

def run_exp(n, p, n_exp = 100):
    """
    Returns two lists 
        - centralities : a list of degree centrality values of the attacker 
        - reconstruction_proportions : a list of the proportion of reconstructed nodes 
    One attacker
    """
    c = 0
    centralities = []
    reconstruction_proportions = []
    while c<n_exp: 
        G = nx.fast_gnp_random_graph(n, p)
        if  nx.is_connected(G):
            c+=1 
            R = Reconstruct(G, n, [0])
            invertible_fraction = R.rank()/n
            centralities.append(nx.eigenvector_centrality(G)[0])
            reconstruction_proportions.append(invertible_fraction)

    return centralities, reconstruction_proportions


def main():
    n = 250
    n_exp = 100
    p_list = [0.04, 0.06, 0.08, 0.1 ]
    ccs = []
    rrs = [] 
    start = time.time()
    times_list = []
    for p in tqdm(p_list):
        cc, rr = run_exp(n, p, n_exp = n_exp)
        ccs.append(cc)
        rrs.append(rr)
        plt.scatter(cc, rr, label = rf'$n = {n}$, $p = {p}$')
        times_list.append(time.time()-start)
    end = time.time()
    plt.legend()
    plt.title("Reconstruction fraction as a function of the centrality of the attacker")
    plt.xlabel("Degree centrality of the attacker")
    plt.ylabel("Fraction of the reconstructed nodes")

    # Creating a folder to save the results 
    exp_dir = f"experiments/centrality_{n}"

    if os.path.isdir(exp_dir):
        exp_dir = exp_dir +'_' + random.choice(string.ascii_letters)
    if os.path.isdir(exp_dir):
        exp_dir = exp_dir + '_' + random.choice(string.ascii_letters)
    os.mkdir(exp_dir)
    print("Created new folder : ", exp_dir)

    # Saving the values
    with open(os.path.join(exp_dir, "res.pkl"), "wb") as f:
        pickle.dump([ccs, rrs], f)
    
    config = {'n_exp' : n_exp ,  'n':  n, 'p_list' : list(p_list) }
    #json_config = json.dumps(config)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(config, f)

    logging.basicConfig(filename=os.path.join(exp_dir, "log.log"), level=logging.INFO)
    logging.info(f"Time taken : {end-start}")
    logging.info(f"Time taken per experiment : {times_list}")
    logging.info("Config : " + str(config))
    logging.info("Centrality type : eigenvector_centrality")
    # Saving the plot
    plt.savefig(os.path.join(exp_dir, "scatter_plot.png")) 

if __name__ == '__main__':
    main()



