
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors as colors
from random import seed

import sys
sys.path.append("src")
from consensus import *

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.rcParams.update({'font.size': 28})

seed(0)
np.random.seed(1)

def process_ego(ego):

    # Same preprocessing as in muffliato
    name_edgelist = "facebook/"+str(ego)+".edges"
    my_graph = nx.read_edgelist(name_edgelist)
    my_graph = nx.relabel_nodes(my_graph, lambda x:int(x))
    Gcc = sorted(nx.connected_components(my_graph), key=len, reverse=True)
    G0 = my_graph.subgraph(Gcc[0]).copy()
    n = G0.number_of_nodes()
    G0 = nx.convert_node_labels_to_integers(G0, label_attribute="fb_id")
    # chose an attacker
    attacker= np.random.randint(n)
    print(ego, n, attacker)

    
    # run the reconstruction
    X = np.random.randint(2, size=n)
    R = Reconstruct(G0, 10, [attacker])
    U, x, x_hat = R.reconstruction(X)
    reconstructible = np.where(np.isnan(x_hat), 0, 1)
    
    colors_a = ['xkcd:silver']*n 
    colors_a[attacker] = 'red'


    return G0, reconstructible, colors_a
    

egos = [0, 348, 414, 686, 698, 3980]

np.random.seed(5) #5

plt.figure(figsize=(10,10))
for i, ego in enumerate(egos):
    G0, colors , attackers = process_ego(ego)
    G0.remove_edges_from(nx.selfloop_edges(G0))
    plt.subplot(3, 2, i+1)
    nx.draw(G0, node_color=colors, node_size=20, alpha=0.5, edgecolors=attackers, edge_color='xkcd:silver', width=.5, cmap=plt.cm.viridis_r)
plt.savefig("allego.pdf", bbox_inches='tight', pad_inches=0)
plt.show()